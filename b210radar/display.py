"""Live micro-Doppler display."""

import time

import numpy as np


class LiveDisplay:
    """Rolling micro-Doppler spectrogram plus a range/speed readout.

    Every frame is accumulated into the waterfall, but the canvas is only
    redrawn at ``max_fps``. A full redraw costs tens of milliseconds, and at
    30.72 MSps the receive loop has little CPU to spare -- drawing on every
    frame is enough to starve it into overflowing.
    """

    def __init__(self, cfg, history=200, max_fps=5.0):
        import matplotlib.pyplot as plt

        self.plt = plt
        self.cfg = cfg
        self.history = history
        self.min_interval = 1.0 / max_fps
        self._last_draw = 0.0
        self._latest = (0.0, False, float("nan"), float("nan"), "")

        # Only keep the speed band that is actually on screen. The Doppler
        # spectrum spans +/- 24 m/s but a hand lives inside +/- 3, so storing
        # and colour-mapping the whole thing would waste ~8x the pixels.
        speeds = 0.5 * cfg.wavelength * np.fft.fftshift(
            np.fft.fftfreq(cfg.frame_len, 1.0 / cfg.slow_rate))
        keep = np.nonzero(np.abs(speeds) <= cfg.search_max_speed)[0]
        self.lo_bin, self.hi_bin = int(keep[0]), int(keep[-1]) + 1
        self.speed_lo, self.speed_hi = speeds[keep[0]], speeds[keep[-1]]

        self.spec = np.full((self.hi_bin - self.lo_bin, history), -120.0)
        self.range_hist = np.full(history, np.nan)

        plt.ion()
        self.fig, (self.ax_s, self.ax_r) = plt.subplots(
            2, 1, figsize=(9.5, 6.5), dpi=90,
            height_ratios=[3, 1], sharex=True)
        self.fig.canvas.manager.set_window_title("B210 two-tone Doppler radar")

        self.im = self.ax_s.imshow(
            self.spec, aspect="auto", origin="lower", cmap="magma",
            extent=[-history / cfg.frame_rate, 0, self.speed_lo, self.speed_hi],
            # The data is already normalised to the background, so noise sits
            # at 0 dB and targets run 10-30 dB above it.
            vmin=-3, vmax=25, interpolation="nearest")
        self.ax_s.set_ylabel("approach speed  (m/s)\n+ = toward radar")
        self.ax_s.axhline(0, color="w", lw=0.5, alpha=0.3)
        self.fig.colorbar(self.im, ax=self.ax_s, label="dB above background")

        self.marker, = self.ax_s.plot([], [], "co", ms=5, mfc="none", mew=1.2)
        self.title = self.ax_s.set_title("starting...", family="monospace")

        self.range_line, = self.ax_r.plot([], [], "c.-", ms=4, lw=1)
        self.ax_r.set_ylim(0, min(cfg.unambiguous_range, 6.0))
        self.ax_r.set_xlim(-history / cfg.frame_rate, 0)
        self.ax_r.set_ylabel("range (m)")
        self.ax_r.set_xlabel("time (s)")
        self.ax_r.grid(alpha=0.3)

        self.fig.tight_layout()

        # Blitting. A full redraw of this figure measures ~110 ms, which at
        # even 8 fps eats most of a core and starves the capture thread into
        # overflows. Marking the four changing artists as animated keeps them
        # out of the cached background, so each frame only re-blits those.
        self._animated = [self.im, self.marker, self.title, self.range_line]
        for art in self._animated:
            art.set_animated(True)
        self._bg = None
        self.fig.canvas.mpl_connect("resize_event", self._invalidate_bg)
        self.fig.canvas.draw()

    def _invalidate_bg(self, _event=None):
        self._bg = None

    def push(self, det, active, range_s, speed_s, status=""):
        """Accumulate one frame. Cheap; safe to call from the capture thread.

        Kept separate from :meth:`render` so that capture never waits on Qt --
        a redraw costs tens of milliseconds and the receive loop cannot afford
        to block for that at 30.72 MSps.
        """
        lo, hi = self.lo_bin, self.hi_bin
        col = 10 * np.log10(np.maximum(det.spectrum[lo:hi], 1e-30)
                            / np.maximum(det.background[lo:hi], 1e-30))
        self.spec = np.roll(self.spec, -1, axis=1)
        self.spec[:, -1] = col

        self.range_hist = np.roll(self.range_hist, -1)
        self.range_hist[-1] = range_s if active else np.nan

        self._latest = (det.snr_db, active, range_s, speed_s, status)

    def render(self, force=False):
        """Redraw the figure, at most ``max_fps`` times a second."""
        now = time.monotonic()
        if not force and now - self._last_draw < self.min_interval:
            return
        self._last_draw = now

        cfg = self.cfg
        self.im.set_data(self.spec)
        t = np.linspace(-self.history / cfg.frame_rate, 0, self.history)
        self.range_line.set_data(t, self.range_hist)

        snr_db, active, range_s, speed_s, status = self._latest
        if active:
            self.marker.set_data([0], [speed_s])
            direction = "CLOSING" if speed_s > 0 else "RECEDING"
            self.title.set_text(
                f"{direction:8s}  range {range_s:5.2f} m   "
                f"speed {abs(speed_s):4.2f} m/s   SNR {snr_db:5.1f} dB   {status}")
            self.title.set_color("tab:cyan" if speed_s > 0 else "tab:orange")
        else:
            self.marker.set_data([], [])
            self.title.set_text(f"{'-- no mover --':40s}   {status}")
            self.title.set_color("0.5")

        canvas = self.fig.canvas
        try:
            if self._bg is None:
                canvas.draw()
                self._bg = canvas.copy_from_bbox(self.fig.bbox)
            canvas.restore_region(self._bg)
            self.ax_s.draw_artist(self.im)
            self.ax_s.draw_artist(self.marker)
            self.ax_s.draw_artist(self.title)
            self.ax_r.draw_artist(self.range_line)
            # Blit only the two axes rather than the whole figure canvas.
            canvas.blit(self.ax_s.bbox)
            canvas.blit(self.ax_r.bbox)
        except Exception:                                    # noqa: BLE001
            # Backend without blitting support: fall back to a full redraw.
            for art in self._animated:
                art.set_animated(False)
            canvas.draw_idle()
        canvas.flush_events()

    @property
    def alive(self):
        return self.plt.fignum_exists(self.fig.number)

    def close(self):
        self.plt.close(self.fig)
