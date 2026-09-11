#!/usr/bin/env python3
"""Measure detection range against target RCS and TX->RX antenna isolation.

Answers the practical question "how far away can this see a hand?", and shows
that the answer is set by antenna isolation rather than by transmit power --
because the limit is LO phase noise beating against the leakage, not thermal
noise.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from b210radar import RadarConfig, RadarProcessor, SimSource, Target  # noqa: E402

RCS = {"hand": 0.01, "head": 0.1, "torso/person": 1.0}


def detect_rate(rng_m, rcs, leakage_db, speed=0.4, seconds=2.0, seed=0):
    """Fraction of frames detecting a target oscillating about `rng_m`."""
    # Shortened settling: this sweep runs hundreds of scenes, and the defaults
    # are tuned for a live session rather than for batch measurement.
    cfg = RadarConfig.sim(clutter_tc=0.3, bg_warmup_frames=25,
                          sim_leakage_db=leakage_db)
    warm = max(3 * cfg.clutter_tc, cfg.bg_warmup_frames / cfg.frame_rate) + 1.0

    # Target moves back and forth at a constant speed around the test range,
    # so it always has a clean Doppler signature. The leg is capped at half the
    # test range: a longer one would swing the target through zero and out the
    # far side of the unambiguous interval.
    leg = min(0.5, 0.5 * rng_m)
    script, t, r, direction = [(0.0, rng_m)], warm, rng_m, -1
    while t < warm + seconds + 2:
        script.append((t, r))
        r += direction * leg
        t += leg / speed
        direction *= -1
    tgt = Target(rcs, script)

    src = SimSource(cfg, targets=[tgt], seed=seed)
    proc = RadarProcessor(cfg)
    dets = []
    for _ in range(int((warm + seconds) * cfg.samp_rate / cfg.fast_block)):
        dets.extend(proc.process(src.recv()))

    rows = [d for d in dets if d.time > warm + cfg.frame_time]
    if not rows:
        return 0.0, float("nan")
    rate = float(np.mean([d.detected for d in rows]))
    hits = [d for d in rows if d.detected]
    err = float(np.median([abs(d.range_m - tgt.state(d.time)[0]) for d in hits])) \
        if hits else float("nan")
    return rate, err


def max_range(rcs, leakage_db, lo=0.2, hi=12.0, steps=5):
    """Largest range still detected in >=80% of frames, found by bisection.

    Detection probability falls monotonically with range (SNR goes as R^-4),
    so bisection is valid and far cheaper than scanning a grid.
    """
    if detect_rate(lo, rcs, leakage_db)[0] < 0.8:
        return 0.0
    if detect_rate(hi, rcs, leakage_db)[0] >= 0.8:
        return hi
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if detect_rate(mid, rcs, leakage_db)[0] >= 0.8:
            lo = mid
        else:
            hi = mid
    return lo


def main():
    cfg = RadarConfig.sim()
    print(cfg.summary())
    print(f"\nphase noise {cfg.sim_phase_noise_dbc_hz:.0f} dBc/Hz @ 100 Hz, "
          f"10 dBm TX, 6 dBi antennas, NF {cfg.sim_noise_figure_db:.0f} dB")
    print("detection = >=80% of frames, target moving at 0.4 m/s\n")

    isolations = [-25, -40, -55]

    print("max detection range vs TX->RX antenna isolation")
    print(f"{'target':>14s} | " + " ".join(f"{i:>7d} dB" for i in isolations))
    print("-" * (14 + 3 + 11 * len(isolations)))
    for name, rcs in RCS.items():
        cells = [f"{max_range(rcs, iso):>7.2f} m" for iso in isolations]
        print(f"{name:>14s} | " + " ".join(cells), flush=True)

    print("\nrange accuracy on a detected hand (two-tone phase):")
    for r in [0.3, 0.6, 1.0, 1.5]:
        rate, err = detect_rate(r, RCS["hand"], -40)
        flag = "" if rate >= 0.8 else "   (below detection threshold)"
        print(f"  true {r:4.2f} m -> median error "
              f"{err*100:5.1f} cm   detect {rate*100:5.1f}%{flag}")


if __name__ == "__main__":
    raise SystemExit(main())
