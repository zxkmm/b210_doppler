"""Sample sources: a real USRP B210, and a physics-based simulator."""

import threading
import time

import numpy as np
from scipy.signal import lfilter

from .config import C

K_BOLTZMANN_DBM_HZ = -174.0   # kT at 290 K, dBm/Hz


def tx_waveform(cfg, n_samples: int) -> np.ndarray:
    """Sum of the transmit tones, scaled so the peak magnitude is tx_amplitude.

    ``n_samples`` must be a multiple of the stage-1 boxcar length so the buffer
    can be replayed back-to-back without a phase discontinuity.
    """
    n = np.arange(n_samples)
    amp = cfg.tx_amplitude / cfg.n_tones
    wf = np.zeros(n_samples, dtype=np.complex64)
    for f in cfg.tone_offsets:
        wf += (amp * np.exp(2j * np.pi * f * n / cfg.samp_rate)).astype(np.complex64)
    return wf


# --------------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------------

class Target:
    """A point scatterer following a piecewise-linear range trajectory."""

    def __init__(self, rcs_m2: float, waypoints):
        self.rcs = float(rcs_m2)
        self.wp = sorted((float(t), float(r)) for t, r in waypoints)
        if len(self.wp) < 1:
            raise ValueError("need at least one waypoint")

    def state(self, t: float):
        """Return ``(range_m, radial_velocity)`` at time ``t``.

        Velocity is positive when receding, matching ``dR/dt``.
        """
        if len(self.wp) == 1 or t <= self.wp[0][0]:
            return self.wp[0][1], 0.0
        if t >= self.wp[-1][0]:
            return self.wp[-1][1], 0.0
        for (t0, r0), (t1, r1) in zip(self.wp, self.wp[1:]):
            if t0 <= t <= t1:
                v = (r1 - r0) / (t1 - t0) if t1 > t0 else 0.0
                return r0 + v * (t - t0), v
        return self.wp[-1][1], 0.0


class SimSource:
    """Synthetic B210 baseband, built from the radar range equation.

    Models: TX->RX antenna leakage, static clutter, moving point targets,
    thermal noise at the stated noise figure, and uncorrelated TX/RX LO phase
    noise.  Amplitudes are referenced to the thermal noise floor over the full
    sampled bandwidth, so SNR numbers coming out of the processor are the ones
    the real link would give.
    """

    def __init__(self, cfg, targets=None, tx_power_dbm=10.0, ant_gain_dbi=6.0,
                 clutter_range=3.0, clutter_rcs=1.0, seed=0, realtime=False):
        self.cfg = cfg
        self.targets = list(targets or [])
        self.tx_power_dbm = tx_power_dbm
        self.ant_gain_dbi = ant_gain_dbi
        self.rng = np.random.default_rng(seed)
        self.realtime = realtime
        self.t = 0.0
        self._t0 = None

        # Noise power in the full sampled bandwidth.
        self.noise_dbm = (K_BOLTZMANN_DBM_HZ + cfg.sim_noise_figure_db
                          + 10 * np.log10(cfg.samp_rate))

        # Per-tone transmit power (equal split).
        self.tone_dbm = tx_power_dbm - 10 * np.log10(cfg.n_tones)

        # Static paths: direct antenna leakage (~zero range) and a wall.
        leak_dbm = tx_power_dbm + cfg.sim_leakage_db
        self.static_paths = [
            (self._amp_from_dbm(leak_dbm), 0.05, 0.0),
            (self._amp_from_rcs(clutter_rcs, clutter_range), clutter_range, 0.0),
        ]

        # Phase-noise shaping: random walk (1/f^2) matched to L(100 Hz),
        # made leaky at 0.1 Hz so the accumulated phase stays bounded.
        h = 2.0 * 10 ** (cfg.sim_phase_noise_dbc_hz / 10.0) * 100.0 ** 2
        self.pn_sigma = np.sqrt(2 * np.pi ** 2 * h / cfg.samp_rate)
        self.pn_pole = float(np.exp(-2 * np.pi * 0.1 / cfg.samp_rate))
        self._pn_zi = [np.zeros(1), np.zeros(1)]

    # -- helpers ---------------------------------------------------------
    def _amp_from_dbm(self, p_dbm):
        """Linear amplitude for a received power, in units of noise sigma."""
        return float(10 ** ((p_dbm - self.noise_dbm) / 20.0))

    def _amp_from_rcs(self, rcs, rng_m):
        """Radar equation: Pr = Pt G^2 lambda^2 sigma / ((4 pi)^3 R^4)."""
        lam = self.cfg.wavelength
        g = 10 ** (self.ant_gain_dbi / 10.0)
        pt = 10 ** ((self.tone_dbm - 30) / 10.0)          # W, per tone
        pr = pt * g * g * lam ** 2 * rcs / ((4 * np.pi) ** 3 * max(rng_m, 1e-3) ** 4)
        pr_dbm = 10 * np.log10(max(pr, 1e-30)) + 30
        return self._amp_from_dbm(pr_dbm)

    def _phase_noise(self, n):
        """Common LO phase psi = phi_tx - phi_rx over the next n samples."""
        psi = np.zeros(n)
        for i in range(2):
            w = self.rng.normal(0.0, self.pn_sigma, n)
            y, self._pn_zi[i] = lfilter([1.0], [1.0, -self.pn_pole], w,
                                        zi=self._pn_zi[i])
            psi += y if i == 0 else -y
        return psi

    def _path_signal(self, n, amp, r0, v):
        """One scatterer's contribution, summed over both transmit tones."""
        cfg = self.cfg
        dt = 1.0 / cfg.samp_rate
        idx = np.arange(n)
        out = np.zeros(n, dtype=np.complex128)
        for f in cfg.tone_offsets:
            fk = cfg.center_freq + f
            # phase(t) = 2 pi f t  -  4 pi (fc+f) (r0 + v t) / c, linear in t
            p0 = -4 * np.pi * fk * r0 / C
            slope = (2 * np.pi * f - 4 * np.pi * fk * v / C) * dt
            # `amp` is already the per-tone amplitude (tone_dbm splits the
            # transmit power), so it must not be divided by n_tones again.
            out += amp * np.exp(1j * (p0 + slope * idx))
        return out

    # -- source interface ------------------------------------------------
    def recv(self, n=None) -> np.ndarray:
        cfg = self.cfg
        n = int(n or cfg.fast_block)
        t0 = self.t

        sig = np.zeros(n, dtype=np.complex128)
        for amp, r0, v in self.static_paths:
            sig += self._path_signal(n, amp, r0, v)
        for tgt in self.targets:
            # Evaluate range at the *start* of the block: the per-block phase
            # ramp begins at index 0, so anything else (e.g. the block centre)
            # leaves a phase discontinuity at every block boundary, which
            # sprays energy across the whole Doppler band.
            r, v = tgt.state(t0)
            sig += self._path_signal(n, self._amp_from_rcs(tgt.rcs, r), r, v)

        sig *= np.exp(1j * self._phase_noise(n))
        noise = (self.rng.normal(0, np.sqrt(0.5), n)
                 + 1j * self.rng.normal(0, np.sqrt(0.5), n))
        self.t += n / cfg.samp_rate

        if self.realtime:
            if self._t0 is None:
                self._t0 = time.monotonic()
            lag = self._t0 + self.t - time.monotonic()
            if lag > 0:
                time.sleep(lag)
        return (sig + noise).astype(np.complex64)

    def start(self):
        return self

    def stop(self):
        pass

    @property
    def status(self):
        return ""


# --------------------------------------------------------------------------
# Real hardware
# --------------------------------------------------------------------------

class UsrpSource:
    """Coherent full-duplex TX/RX on a USRP B210.

    The B210's TX and RX chains sit on one AD9361 driven by a single reference,
    so the two run at exactly the same frequency with a fixed (if arbitrary)
    phase offset -- which is all a Doppler radar needs.  TX and RX are started
    at a common ``time_spec`` so the streams stay index-aligned.

    Antennas: transmit on ``TX/RX``, receive on ``RX2``.  Separate them as far
    as the cabling allows and cross-polarise them if you can; every dB of
    TX->RX isolation buys a dB of phase-noise floor (see README).
    """

    def __init__(self, cfg, device_args=""):
        import uhd  # imported lazily so the simulator works without UHD

        self.uhd = uhd
        self.cfg = cfg
        args = device_args or ""
        # Deep USB buffers absorb scheduler jitter; without them a single late
        # wake-up of the Python receive loop costs an overflow. 256 frames per
        # direction is roughly 8 MB at UHD's USB3 frame size, which fits the
        # usual 16 MB usbfs limit -- asking for much more returns
        # LIBUSB_ERROR_NO_MEM. Raising
        # /sys/module/usbcore/parameters/usbfs_memory_mb allows more.
        # Frame *sizes* are left alone: UHD's USB3 default is larger than
        # anything worth hand-picking here.
        # Weighted towards receive: matplotlib's Agg renderer holds the GIL for
        # tens of milliseconds at a time, and the receive buffer has to be deep
        # enough to ride that out. 512 recv + 128 send frames is ~10 MB at
        # UHD's USB3 frame size, leaving room under the usual 16 MB cap.
        defaults = {
            "master_clock_rate": cfg.master_clock_rate,
            "num_recv_frames": 512,
            "num_send_frames": 128,
        }
        for key, value in defaults.items():
            if key not in args:
                args = f"{args},{key}={value}" if args else f"{key}={value}"
        self.usrp = uhd.usrp.MultiUSRP(args)

        rx_chans = list(cfg.rx_channels)
        self.rx_chans = rx_chans

        self.usrp.set_tx_rate(cfg.samp_rate, 0)
        self.usrp.set_tx_freq(uhd.types.TuneRequest(cfg.center_freq), 0)
        self.usrp.set_tx_gain(cfg.tx_gain, 0)
        self.usrp.set_tx_antenna(cfg.tx_antenna, 0)
        self.usrp.set_tx_bandwidth(cfg.samp_rate, 0)

        for ch in rx_chans:
            self.usrp.set_rx_rate(cfg.samp_rate, ch)
            self.usrp.set_rx_freq(uhd.types.TuneRequest(cfg.center_freq), ch)
            self.usrp.set_rx_gain(cfg.rx_gain, ch)
            self.usrp.set_rx_antenna(cfg.rx_antenna, ch)
            self.usrp.set_rx_bandwidth(cfg.samp_rate, ch)
            # Automatic DC-offset and IQ-balance correction are nice to have,
            # not required: the tones are deliberately offset from DC, so the
            # residuals land outside the band the downconverter keeps. Some
            # B210 clones (e.g. LibreSDR B220mini) reject these calls.
            for setter in (self.usrp.set_rx_dc_offset,
                           self.usrp.set_rx_iq_balance):
                try:
                    setter(True, ch)
                except Exception:                            # noqa: BLE001
                    pass

        self.actual_rate = self.usrp.get_rx_rate(rx_chans[0])
        if abs(self.actual_rate - cfg.samp_rate) > 1.0:
            raise RuntimeError(
                f"B210 gave {self.actual_rate/1e6:.6f} MSps, not the requested "
                f"{cfg.samp_rate/1e6:.6f} MSps; the tone/decimation constraints "
                f"assume the exact rate. Pick a rate that divides "
                f"master_clock_rate={cfg.master_clock_rate/1e6:.3f} MHz."
            )

        tx_args = uhd.usrp.StreamArgs("fc32", "sc16")
        tx_args.channels = [0]
        self.tx_streamer = self.usrp.get_tx_stream(tx_args)

        rx_args = uhd.usrp.StreamArgs("fc32", "sc16")
        rx_args.channels = rx_chans
        self.rx_streamer = self.usrp.get_rx_stream(rx_args)

        self.block = cfg.fast_block
        self.tx_buf = tx_waveform(cfg, self.block).reshape(1, -1)
        self.rx_buf = np.zeros((len(rx_chans), self.block), dtype=np.complex64)
        self.rx_md = uhd.types.RXMetadata()

        self._tx_thread = None
        self._running = threading.Event()
        self.overflows = 0
        self.underflows = 0

    def start(self, lead_time=0.5):
        uhd = self.uhd
        self.usrp.set_time_now(uhd.types.TimeSpec(0.0))
        t_start = lead_time

        self._running.set()
        self._tx_thread = threading.Thread(
            target=self._tx_loop, args=(t_start,), daemon=True)
        self._tx_thread.start()

        cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        cmd.stream_now = False
        cmd.time_spec = uhd.types.TimeSpec(t_start)
        self.rx_streamer.issue_stream_cmd(cmd)
        return self

    def _tx_loop(self, t_start):
        uhd = self.uhd
        md = uhd.types.TXMetadata()
        md.start_of_burst = True
        md.end_of_burst = False
        md.has_time_spec = True
        md.time_spec = uhd.types.TimeSpec(t_start)
        while self._running.is_set():
            self.tx_streamer.send(self.tx_buf, md, 1.0)
            md.start_of_burst = False
            md.has_time_spec = False
        md.end_of_burst = True
        self.tx_streamer.send(np.zeros((1, 1), dtype=np.complex64), md, 1.0)

    def recv(self, n=None) -> np.ndarray:
        """Return ``block`` samples from RX channel 0 (or the referenced product)."""
        got = self.rx_streamer.recv(self.rx_buf, self.rx_md, 1.0)
        err = self.rx_md.error_code
        if err == self.uhd.types.RXMetadataErrorCode.overflow:
            self.overflows += 1
        elif err != self.uhd.types.RXMetadataErrorCode.none:
            raise RuntimeError(f"RX error: {self.rx_md.strerror()}")
        data = self.rx_buf[:, :got]

        if len(self.rx_chans) >= 2:
            # Optional coupler reference on channel 1. Both RX chains share the
            # AD9361 RX synthesiser and see the same TX, so this conjugate
            # product cancels TX *and* RX LO phase noise exactly.
            ref = data[1]
            mag2 = np.maximum(np.abs(ref) ** 2, 1e-20)
            return (data[0] * np.conj(ref) / mag2).astype(np.complex64)
        return data[0].copy()

    def stop(self):
        self._running.clear()
        if self._tx_thread is not None:
            self._tx_thread.join(timeout=2.0)
        cmd = self.uhd.types.StreamCMD(self.uhd.types.StreamMode.stop_cont)
        self.rx_streamer.issue_stream_cmd(cmd)

    @property
    def status(self):
        bits = []
        if self.overflows:
            bits.append(f"overflows={self.overflows}")
        if self.underflows:
            bits.append(f"underflows={self.underflows}")
        return "  ".join(bits)
