"""Streaming DSP blocks: tone downconversion, decimation, clutter cancellation."""

import numpy as np
from scipy.signal import firwin, lfilter


class FirDecimator:
    """Stateful FIR low-pass + integer decimation for a continuous stream.

    Carries both the filter state and the decimation phase across blocks, so
    feeding the input in arbitrary-length chunks gives exactly the same output
    as filtering the whole stream at once.
    """

    def __init__(self, factor: int, taps_per_phase: int = 8):
        self.factor = int(factor)
        numtaps = taps_per_phase * self.factor + 1
        self.taps = firwin(numtaps, 0.8 / self.factor).astype(np.float64)
        self.zi = np.zeros(len(self.taps) - 1, dtype=np.complex128)
        self.phase = 0  # count of samples consumed so far, mod factor

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        y, self.zi = lfilter(self.taps, [1.0], x, zi=self.zi)
        start = (-self.phase) % self.factor
        self.phase = (self.phase + x.size) % self.factor
        return y[start::self.factor]


class ToneDownconverter:
    """Mix a wideband stream down to one slow stream per transmitted tone.

    Stage 1 multiplies by the tone LOs and sums over ``boxcar_decims[0]``
    samples in a single complex GEMM.  This is only valid because each tone
    completes a whole number of cycles per block (enforced by
    :meth:`RadarConfig.validate`), which makes the mixer table identical for
    every block.  Stage 2 is a second boxcar, giving sinc^2 (CIC) alias
    rejection, and the remaining decimation is done with proper FIR stages at a
    rate low enough that their cost is negligible.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        d1, d2 = cfg.boxcar_decims
        self.d1, self.d2 = int(d1), int(d2)
        n = np.arange(self.d1)
        tones = np.asarray(cfg.tone_offsets, dtype=np.float64)
        # columns are conj(LO) for each tone, scaled to unity DC gain
        w = np.exp(-2j * np.pi * np.outer(n, tones) / cfg.samp_rate) / self.d1
        self.mixer = np.ascontiguousarray(w.astype(np.complex64))
        self.fir = [[FirDecimator(f) for f in cfg.fir_decims]
                    for _ in range(cfg.n_tones)]
        self._tail1 = np.zeros(0, dtype=np.complex64)
        self._tail2 = np.zeros((0, cfg.n_tones), dtype=np.complex64)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Consume fast-time samples, return shape ``(n_tones, n_slow)``.

        Any samples left over from a partial stage-1 block are carried to the
        next call. UHD's ``recv`` is free to return short reads, and silently
        dropping the remainder would slip slow-time phase.
        """
        if self._tail1.size:
            x = np.concatenate([self._tail1, x])
        n = (x.size // self.d1) * self.d1
        self._tail1 = x[n:].copy()
        if n == 0:
            return np.zeros((self.cfg.n_tones, 0), dtype=np.complex64)
        y = x[:n].reshape(-1, self.d1) @ self.mixer           # (blocks, tones)

        # second boxcar stage, carrying a partial block between calls
        y = np.concatenate([self._tail2, y], axis=0)
        m = (y.shape[0] // self.d2) * self.d2
        self._tail2 = y[m:]
        if m == 0:
            return np.zeros((self.cfg.n_tones, 0), dtype=np.complex64)
        y = y[:m].reshape(-1, self.d2, self.cfg.n_tones).mean(axis=1)

        out = []
        for k in range(self.cfg.n_tones):
            v = y[:, k].astype(np.complex128)
            for stage in self.fir[k]:
                v = stage(v)
            out.append(v)
        width = min(len(v) for v in out)
        return np.stack([v[:width] for v in out], axis=0).astype(np.complex64)


class ClutterCanceller:
    """Remove static returns (TX/RX leakage, walls, furniture).

    Tracks a slowly-adapting estimate of the stationary component with a
    one-pole IIR and subtracts it, which is a high-pass on slow time.  The
    running estimate is also exposed, because its two-tone phase difference is
    exactly the hardware delay calibration.
    """

    def __init__(self, cfg):
        self.alpha = 1.0 - np.exp(-1.0 / (cfg.clutter_tc * cfg.slow_rate))
        self.state = np.zeros(cfg.n_tones, dtype=np.complex128)
        self._primed = False

    @property
    def static(self) -> np.ndarray:
        return self.state.copy()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """``x`` is ``(n_tones, n)``; returns the moving-target component."""
        if x.shape[1] == 0:
            return x
        if not self._primed:
            self.state = x[:, 0].astype(np.complex128).copy()
            self._primed = True
        b = [self.alpha]
        a = [1.0, -(1.0 - self.alpha)]
        out = np.empty_like(x)
        for k in range(x.shape[0]):
            zi = [self.state[k] * (1.0 - self.alpha)]
            est, zf = lfilter(b, a, x[k].astype(np.complex128), zi=zi)
            self.state[k] = zf[0] / (1.0 - self.alpha)
            out[k] = (x[k] - est).astype(np.complex64)
        return out


class SlowTimeBuffer:
    """Sliding window over the slow-time stream, emitting overlapping frames."""

    def __init__(self, cfg):
        self.frame_len = cfg.frame_len
        self.hop = cfg.frame_hop
        self.buf = np.zeros((cfg.n_tones, cfg.frame_len), dtype=np.complex64)
        self.filled = 0
        self.since_last = 0
        self.n_seen = 0

    def push(self, x: np.ndarray):
        """Append samples; yield ``(index_of_newest_sample, frame)`` per hop.

        The input is consumed in pieces no longer than one hop, so a caller
        feeding large chunks (a recorded file, say) gets the same sequence of
        frames as one feeding a couple of samples at a time.
        """
        n = x.shape[1]
        pos = 0
        while pos < n:
            take = min(n - pos, self.hop - self.since_last)
            chunk = x[:, pos:pos + take]
            k = chunk.shape[1]
            self.buf = np.roll(self.buf, -k, axis=1)
            self.buf[:, -k:] = chunk
            self.filled = min(self.filled + k, self.frame_len)
            self.n_seen += k
            self.since_last += k
            pos += take

            if self.since_last >= self.hop:
                # Reset unconditionally: if the buffer is not full yet there is
                # no frame to emit, but leaving the counter at the hop would
                # make `take` zero and spin forever.
                self.since_last = 0
                if self.filled >= self.frame_len:
                    yield self.n_seen, self.buf.copy()


def wrap_0_2pi(phi):
    """Wrap radians into [0, 2*pi)."""
    return np.mod(phi, 2.0 * np.pi)


def wrap_pi(phi):
    """Wrap radians into (-pi, pi]."""
    return np.mod(phi + np.pi, 2.0 * np.pi) - np.pi
