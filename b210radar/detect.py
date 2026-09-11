"""Doppler + two-tone-range detection of moving scatterers (e.g. a hand)."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

from .config import C
from .dsp import ClutterCanceller, SlowTimeBuffer, ToneDownconverter, wrap_0_2pi


@dataclass
class Detection:
    time: float              # s since start of stream
    detected: bool
    snr_db: float
    approach_speed: float    # m/s, positive means moving toward the radar
    range_m: float           # m, absolute range from the two-tone phase
    range_valid: bool
    phase_diff: float        # rad, raw two-tone phase difference before calibration
    amplitude: float         # linear magnitude of the peak Doppler bin
    spectrum: np.ndarray     # |X|^2 of tone 0, fftshifted, clutter-suppressed
    background: np.ndarray   # CFAR background estimate, same axis
    speeds: np.ndarray       # approach-speed axis matching `spectrum`


class RadarProcessor:
    """Full chain: downconvert -> clutter cancel -> Doppler FFT -> detect.

    Sign conventions are fixed by the baseband model of a scatterer at range
    ``R(t)`` illuminated by tone ``f_k`` on carrier ``f_c``::

        z_k(t) = a * exp(-j * 4*pi * (f_c + f_k) * R(t) / c)

    so a *decreasing* range (approaching) produces a *positive* Doppler
    frequency, and the phase difference between the tones is::

        arg(z_0 * conj(z_1)) = 4*pi * delta_f * R / c

    which inverts to an absolute range that is unambiguous out to
    ``c / (2 * delta_f)``.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.dnc = ToneDownconverter(cfg)
        self.clutter = ClutterCanceller(cfg)
        self.frames = SlowTimeBuffer(cfg)
        self.window = np.hanning(cfg.frame_len).astype(np.float64)
        # Window loss, so `amplitude` is comparable to the input magnitude.
        self.win_gain = self.window.sum()

        f = np.fft.fftshift(np.fft.fftfreq(cfg.frame_len, 1.0 / cfg.slow_rate))
        self.doppler_hz = f
        # positive Doppler == approaching, so approach speed is +lambda/2 * f_d
        self.speeds = 0.5 * cfg.wavelength * f
        self.dc_bin = int(np.argmin(np.abs(f)))

        # Greatest-of CFAR. The background is not flat: LO phase noise beating
        # against the TX->RX leakage puts a steep 1/f^2 pedestal around zero
        # Doppler. A centred estimator is biased low on that slope and
        # false-alarms along it, so the background is taken as the larger of
        # two one-sided training medians -- GO-CFAR, which exists for exactly
        # this clutter-edge case.
        self.cfar_train = max(9, (cfg.frame_len // 16) | 1)
        self.cfar_guard = 3
        self.cfar_offset = self.cfar_guard + 1 + self.cfar_train // 2

        # Spatial CFAR alone cannot cope with the inner skirt of the pedestal:
        # on a steep monotone edge every training cell is further from the
        # clutter than the cell under test, so the background is always biased
        # low. The pedestal is, however, *temporally* stationary while a hand
        # is not -- so the primary background is a per-bin running median
        # tracked across frames (a Robbins-Monro sign update, which converges
        # to the median and is insensitive to the occasional target). The
        # spatial estimate is kept as a floor so a newly-appearing stationary
        # interferer cannot mask targets while the tracker catches up.
        self.bg_track = None
        self.bg_mu = cfg.bg_track_mu
        self.bg_frames = 0
        self.bg_warmup = cfg.bg_warmup_frames

        # Bins too slow / too fast to be a hand, as offsets either side of DC.
        bin_hz = cfg.slow_rate / cfg.frame_len
        self.min_bin = max(1, int(np.ceil(2.0 * cfg.min_speed
                                          / cfg.wavelength / bin_hz)))
        self.max_bin = min(cfg.frame_len // 2 - 1,
                           int(2.0 * cfg.search_max_speed / cfg.wavelength / bin_hz))

        # Search only the speed band a hand can plausibly occupy. Narrowing it
        # also keeps the decimation filter's roll-off at the band edges out of
        # the detector.
        self.search_mask = np.zeros(cfg.frame_len)
        for sgn in (-1, 1):
            a = self.dc_bin + sgn * self.min_bin
            b = self.dc_bin + sgn * self.max_bin
            self.search_mask[min(a, b):max(a, b) + 1] = 1.0

    def _cfar_background(self, power: np.ndarray) -> np.ndarray:
        med = median_filter(power, size=self.cfar_train, mode="nearest")
        n = power.size
        idx = np.arange(n)
        left = med[np.clip(idx - self.cfar_offset, 0, n - 1)]
        right = med[np.clip(idx + self.cfar_offset, 0, n - 1)]
        return np.maximum(np.maximum(left, right), 1e-30)

    # -- streaming interface -------------------------------------------
    def process(self, samples: np.ndarray):
        """Feed fast-time samples; yield a :class:`Detection` per frame."""
        slow = self.dnc(samples)
        if slow.shape[1] == 0:
            return
        moving = self.clutter(slow)
        for n_seen, frame in self.frames.push(moving):
            # Timestamp the frame at its centre: speed and range are averages
            # over the coherent window, so attributing them to the window's
            # end would bias range by v * frame_time / 2.
            centre = n_seen - self.cfg.frame_len / 2.0
            yield self._analyse(centre / self.cfg.slow_rate, frame)

    # -- per-frame analysis ---------------------------------------------
    def _analyse(self, t: float, frame: np.ndarray) -> Detection:
        cfg = self.cfg
        spec = np.fft.fftshift(
            np.fft.fft(frame * self.window, axis=1), axes=1) / self.win_gain
        power = np.abs(spec[0]) ** 2

        # Background = temporal per-bin median, floored by the spatial GO-CFAR.
        if self.bg_track is None:
            self.bg_track = np.maximum(power.copy(), 1e-30)
        background = np.maximum(self._cfar_background(power), self.bg_track)
        ratio = power / background * self.search_mask

        peak = int(np.argmax(ratio))
        snr_db = 10.0 * np.log10(max(ratio[peak], 1e-30))
        self.bg_frames += 1
        detected = bool(snr_db >= cfg.detect_snr_db
                        and self.bg_frames > self.bg_warmup)

        # Advance the tracker, but freeze the bins holding a live detection so
        # a hand moving at a steady speed is not slowly absorbed into its own
        # background.
        step = np.full(power.shape, -self.bg_mu)
        step[power > self.bg_track] = self.bg_mu
        if detected:
            lo = max(peak - self.cfar_guard, 0)
            step[lo:peak + self.cfar_guard + 1] = 0.0
        self.bg_track = np.maximum(self.bg_track * np.exp(step), 1e-30)

        dphi = np.angle(spec[0, peak] * np.conj(spec[1, peak]))
        rng = C * wrap_0_2pi(dphi - cfg.range_phase_offset) / (4.0 * np.pi * cfg.delta_f)

        return Detection(
            time=t,
            detected=detected,
            snr_db=float(snr_db),
            approach_speed=float(self.speeds[peak]),
            range_m=float(rng),
            range_valid=detected,
            phase_diff=float(dphi),
            amplitude=float(np.abs(spec[0, peak])),
            spectrum=power,
            background=background,
            speeds=self.speeds,
        )

    # -- calibration -----------------------------------------------------
    def leakage_phase(self) -> float:
        """Two-tone phase difference of the static scene.

        The TX->RX leakage dominates the stationary return and sits at
        essentially zero range, so this phase is *approximately* the fixed
        hardware delay through cables, filters and converters.  Assign it to
        ``cfg.range_phase_offset`` to put true range zero at range zero.

        It is only approximate: the estimate is the vector sum of everything
        stationary, so a strong nearby wall pulls it.  In testing, a 3 m wall
        26 dB below the leakage biased the range scale by 11 cm.  Use it for a
        quick zero with a clear field of view; use
        :func:`phase_offset_for_known_range` when you need the accuracy.
        """
        s = self.clutter.static
        return float(np.angle(s[0] * np.conj(s[1])))


def phase_offset_for_known_range(detections, true_range_m, delta_f) -> float:
    """Calibration offset from a mover observed at a known range.

    Wave a hand (or a corner reflector) at a measured distance, collect the
    detections, and pass them here.  This references the scale to a real echo
    rather than to the static scene, so it is immune to the wall-pull that
    limits :meth:`RadarProcessor.leakage_phase`.
    """
    hits = [d for d in detections if d.detected]
    if not hits:
        raise ValueError("no detections to calibrate against")
    want = 4.0 * np.pi * delta_f * true_range_m / C
    # Average the residual as a unit vector: it is an angle, so a plain mean
    # would break across the +/-pi wrap.
    resid = np.array([d.phase_diff for d in hits]) - want
    return float(np.angle(np.mean(np.exp(1j * resid))))


class TrackSmoother:
    """Light temporal smoothing and hysteresis over per-frame detections.

    Frame-to-frame estimates are noisy near the detection threshold; this
    holds a detection for ``hold`` frames and median-filters the range so the
    reported track does not flicker.
    """

    def __init__(self, hold: int = 5, history: int = 7):
        self.hold = hold
        self.history = history
        self._miss = hold
        self._ranges = []
        self._speeds = []

    def update(self, det: Detection):
        if det.detected:
            self._miss = 0
            self._ranges.append(det.range_m)
            self._speeds.append(det.approach_speed)
            del self._ranges[:-self.history]
            del self._speeds[:-self.history]
        else:
            self._miss += 1
            if self._miss >= self.hold:
                self._ranges.clear()
                self._speeds.clear()

        active = self._miss < self.hold and bool(self._ranges)
        if not active:
            return False, float("nan"), float("nan")
        return (True,
                float(np.median(self._ranges)),
                float(np.median(self._speeds)))
