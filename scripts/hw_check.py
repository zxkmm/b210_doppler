#!/usr/bin/env python3
"""Hardware smoke test: confirm the B210 transmits and receives coherently.

Streams for a few seconds and checks that the two transmitted tones show up in
the received spectrum at the expected offsets.  If they do, the TX chain, the
RX chain and the antenna coupling are all working and the radar can run.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from b210radar import RadarConfig, UsrpSource, with_overrides  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--freq", type=float)
    p.add_argument("--tx-gain", type=float)
    p.add_argument("--rx-gain", type=float)
    p.add_argument("--samp-rate", type=float)
    p.add_argument("--device-args", default="")
    p.add_argument("--seconds", type=float, default=3.0)
    a = p.parse_args()

    cfg = with_overrides(RadarConfig.hardware(), center_freq=a.freq,
                         tx_gain=a.tx_gain, rx_gain=a.rx_gain,
                         samp_rate=a.samp_rate)
    cfg.validate()
    print(cfg.summary())
    print()

    src = UsrpSource(cfg, device_args=a.device_args)
    print(f"actual rx rate {src.actual_rate/1e6:.6f} MSps, "
          f"block {src.block} samples\n")
    src.start()
    try:
        n = int(a.seconds * cfg.samp_rate / cfg.fast_block)
        blocks, clipped = [], 0
        for i in range(n):
            x = src.recv()
            clipped += int(np.sum(np.abs(x.real) > 0.99) +
                           np.sum(np.abs(x.imag) > 0.99))
            if i >= n // 2 and len(blocks) < 8:
                blocks.append(x)
        x = np.concatenate(blocks)
    finally:
        src.stop()

    print(f"received {x.size} samples   rms {np.sqrt(np.mean(np.abs(x)**2)):.4f}"
          f"   peak {np.max(np.abs(x)):.4f}")
    if clipped:
        print(f"  WARNING: {clipped} clipped samples -- reduce --rx-gain")
    print(f"  stream issues: {src.status or 'none'}")

    # Spectrum of one block; the tones must sit at the commanded offsets.
    seg = x[:cfg.fast_block] * np.hanning(cfg.fast_block)
    spec = np.abs(np.fft.fftshift(np.fft.fft(seg))) ** 2
    freqs = np.fft.fftshift(np.fft.fftfreq(cfg.fast_block, 1 / cfg.samp_rate))
    floor = np.median(spec)

    print("\n  offset        expected      measured       SNR")
    ok = True
    for f in cfg.tone_offsets:
        k = int(np.argmin(np.abs(freqs - f)))
        win = spec[max(k - 3, 0):k + 4]
        snr = 10 * np.log10(win.max() / floor)
        peak_at = freqs[max(k - 3, 0) + int(np.argmax(win))]
        good = snr > 25
        ok &= good
        print(f"  tone       {f/1e6:+8.2f} MHz  {peak_at/1e6:+8.2f} MHz  "
              f"{snr:6.1f} dB  {'OK' if good else 'NOT FOUND'}")

    k0 = int(np.argmin(np.abs(freqs)))
    print(f"  DC/LO leak    0.00 MHz              "
          f"{10*np.log10(spec[k0]/floor):6.1f} dB")

    print()
    if ok:
        print("TX -> RX path confirmed. Run:  python3 scripts/radar.py --usrp")
    else:
        print("Tones not found. Check that antennas are on TX/RX and RX2, and\n"
              "that --tx-gain is high enough.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
