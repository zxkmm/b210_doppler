"""Radar configuration and derived RF/DSP quantities."""

from dataclasses import dataclass, replace

import numpy as np

C = 299_792_458.0


@dataclass
class RadarConfig:
    """Parameters for a two-tone (FSK-CW) Doppler radar on a USRP B210.

    The transmit waveform is the sum of two baseband tones at ``tone_offsets``,
    so the radiated spectrum is two CW carriers at ``center_freq + offset``.
    Doppler processing isolates moving scatterers; the phase difference between
    the two tones then yields the absolute range of the dominant mover.
    """

    # --- RF ---
    center_freq: float = 5.80e9
    samp_rate: float = 30.72e6
    master_clock_rate: float = 61.44e6
    tone_offsets: tuple = (1.92e6, 11.52e6)
    tx_gain: float = 70.0          # B210 TX gain range is 0 .. 89.75 dB
    # Measured on a LibreSDR B220mini: 60 dB puts the leakage at about
    # -22 dBFS rms with no clipping and no overflows. 70 dB gains another 3 dB
    # of tone SNR but starts to clip, and there is little point pushing past
    # the ADC noise floor when the limit is the phase-noise pedestal anyway.
    rx_gain: float = 60.0          # B210 RX gain range is 0 .. 76 dB
    tx_antenna: str = "TX/RX"
    rx_antenna: str = "RX2"
    tx_amplitude: float = 0.3      # peak baseband magnitude, keep headroom for IM
    rx_channels: tuple = (0,)      # (0,) = normal; (0, 1) = ch1 is a coupler reference

    block_ms: float = 8.0          # milliseconds of signal per RX read

    # --- decimation chain ---
    # Two boxcar stages (a CIC, sinc^2 alias rejection) then two FIR stages.
    boxcar_decims: tuple = (16, 16)
    fir_decims: tuple = (8, 8)

    # --- slow-time processing ---
    frame_len: int = 512           # Doppler FFT length
    frame_hop: int = 128           # samples between successive frames
    clutter_tc: float = 2.0        # s, time constant of the static-clutter canceller
    min_speed: float = 0.10        # m/s, slower movers are treated as clutter
    search_max_speed: float = 3.0  # m/s, faster movers are not a hand
    detect_snr_db: float = 12.0    # dB above the local CFAR background
    bg_track_mu: float = 0.05      # per-frame step of the per-bin median tracker
    bg_warmup_frames: int = 40     # frames before detections are trusted

    # --- calibration ---
    # Round-trip phase through cables/filters/converters, measured on the
    # TX->RX leakage path (which is at essentially zero range). Subtracted
    # from the two-tone phase difference before converting to range.
    range_phase_offset: float = 0.0

    # --- simulation only ---
    # LO phase noise at ~100 Hz offset, per synthesiser, assumed 1/f^2.
    sim_phase_noise_dbc_hz: float = -70.0
    # TX->RX antenna isolation. This is THE range-limiting parameter: LO phase
    # noise beating against the leakage sets a pedestal around zero Doppler,
    # and it scales dB-for-dB with this number. -40 dB is what two separated,
    # cross-polarised antennas give; -25 dB is two patches side by side.
    sim_leakage_db: float = -40.0
    sim_noise_figure_db: float = 6.0

    def __post_init__(self):
        self.validate()

    # ---- presets -------------------------------------------------------
    @classmethod
    def hardware(cls, **kw):
        """Full-rate profile for a real B210 over USB 3.0."""
        return cls(**kw)

    @classmethod
    def sim(cls, **kw):
        """Quarter-rate profile so the simulator runs near real time.

        Keeps every constraint of the hardware profile (integer tone cycles
        per boxcar block, same decimation structure), just scaled down.
        """
        base = dict(
            samp_rate=7.68e6,
            master_clock_rate=30.72e6,
            tone_offsets=(0.48e6, 2.88e6),
            # The slow rate is 4x lower, so shorten the frame to keep the
            # coherent integration time near 0.27 s -- long enough for fine
            # Doppler resolution, short enough that a hand's speed is roughly
            # constant across the frame.
            frame_len=128,
            frame_hop=32,
        )
        base.update(kw)
        return cls(**base)

    # ---- derived quantities -------------------------------------------
    @property
    def n_tones(self) -> int:
        return len(self.tone_offsets)

    @property
    def boxcar_total(self) -> int:
        return int(np.prod(self.boxcar_decims))

    @property
    def total_decim(self) -> int:
        return self.boxcar_total * int(np.prod(self.fir_decims))

    @property
    def slow_rate(self) -> float:
        """Sample rate of the slow-time (post-decimation) stream, Hz."""
        return self.samp_rate / self.total_decim

    @property
    def delta_f(self) -> float:
        return self.tone_offsets[1] - self.tone_offsets[0]

    @property
    def wavelength(self) -> float:
        return C / (self.center_freq + float(np.mean(self.tone_offsets)))

    @property
    def unambiguous_range(self) -> float:
        """Range beyond which the two-tone phase difference wraps, m."""
        return C / (2.0 * self.delta_f)

    @property
    def max_speed(self) -> float:
        """Largest unaliased radial speed, m/s."""
        return self.slow_rate * self.wavelength / 4.0

    @property
    def speed_resolution(self) -> float:
        """Width of one Doppler bin expressed as radial speed, m/s."""
        return self.wavelength * self.slow_rate / (2.0 * self.frame_len)

    @property
    def frame_time(self) -> float:
        """Coherent integration time of one Doppler frame, s."""
        return self.frame_len / self.slow_rate

    @property
    def frame_rate(self) -> float:
        return self.slow_rate / self.frame_hop

    @property
    def fast_block(self) -> int:
        """Samples per RX read; a multiple of the boxcar product.

        Bigger is better on hardware: every read and every transmit buffer
        costs a Python call and a GIL handover, and at 1 ms blocks the two
        streams together make ~2000 of them a second, which is enough to
        starve the receiver into overflowing.
        """
        n = self.samp_rate * self.block_ms * 1e-3 / self.boxcar_total
        return max(1, int(round(n))) * self.boxcar_total

    # ---- validation ----------------------------------------------------
    def validate(self):
        if self.n_tones != 2:
            raise ValueError("exactly two tones are required for FSK ranging")
        first = self.boxcar_decims[0]
        for f in self.tone_offsets:
            cycles = f * first / self.samp_rate
            if abs(cycles - round(cycles)) > 1e-9:
                raise ValueError(
                    f"tone {f/1e6:.3f} MHz must complete a whole number of cycles per "
                    f"{first}-sample boxcar block at {self.samp_rate/1e6:.3f} MSps "
                    f"(got {cycles:.6f}); the stage-1 mixer table is tiled and assumes this"
                )
            if abs(f) >= self.samp_rate / 2:
                raise ValueError(f"tone {f/1e6:.3f} MHz exceeds Nyquist")
        if self.delta_f <= 0:
            raise ValueError("tone_offsets must be increasing")
        if self.frame_hop > self.frame_len:
            raise ValueError("frame_hop must not exceed frame_len")

    def summary(self) -> str:
        lines = [
            f"carrier            {self.center_freq/1e9:.4f} GHz  (lambda = {self.wavelength*100:.2f} cm)",
            f"tones              {self.tone_offsets[0]/1e6:+.2f} / {self.tone_offsets[1]/1e6:+.2f} MHz"
            f"  (delta_f = {self.delta_f/1e6:.2f} MHz)",
            f"sample rate        {self.samp_rate/1e6:.3f} MSps  ->  slow rate {self.slow_rate:.1f} Hz"
            f"  (decim {self.total_decim})",
            f"unambiguous range  {self.unambiguous_range:.2f} m",
            f"max radial speed   +/- {self.max_speed:.2f} m/s",
            f"speed resolution   {self.speed_resolution*100:.2f} cm/s"
            f"  ({self.frame_time*1e3:.0f} ms coherent integration)",
            f"frame rate         {self.frame_rate:.1f} Hz",
        ]
        return "\n".join(lines)


def with_overrides(cfg: RadarConfig, **kw) -> RadarConfig:
    """Return a copy of ``cfg`` with the given fields replaced."""
    return replace(cfg, **{k: v for k, v in kw.items() if v is not None})
