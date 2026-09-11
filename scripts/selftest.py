#!/usr/bin/env python3
"""End-to-end validation of the radar chain against a simulated hand.

Runs the exact DSP used on hardware over a synthetic scene with known ground
truth, and checks detection rate, Doppler sign, speed accuracy and absolute
two-tone range accuracy.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from b210radar import C  # noqa: E402
from b210radar import (RadarConfig, RadarProcessor, SimSource,  # noqa: E402
                       Target, TrackSmoother)

HAND_RCS = 0.01          # m^2, a typical open hand at 5-6 GHz


def run(cfg, targets, duration, seed=0, calibrate=True):
    src = SimSource(cfg, targets=targets, seed=seed)
    proc = RadarProcessor(cfg)

    if calibrate:
        # Settle the clutter canceller on the static scene, then zero the
        # hardware delay using the leakage path, exactly as on hardware.
        cal_cfg = RadarConfig(**{**cfg.__dict__})
        cal_src = SimSource(cal_cfg, targets=[], seed=seed + 1)
        cal_proc = RadarProcessor(cal_cfg)
        for _ in range(int(3 * cal_cfg.clutter_tc * cal_cfg.samp_rate
                           / cal_cfg.fast_block)):
            list(cal_proc.process(cal_src.recv()))
        cfg.range_phase_offset = cal_proc.leakage_phase()
        proc = RadarProcessor(cfg)

    n_blocks = int(duration * cfg.samp_rate / cfg.fast_block)
    dets = []
    for _ in range(n_blocks):
        dets.extend(proc.process(src.recv()))
    return dets


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return ok


def check_streaming(cfg):
    """Ragged short reads must give bit-identical output to one big read.

    UHD's recv() is free to return fewer samples than asked for, and this is
    the one class of bug the simulator cannot surface on its own -- SimSource
    always returns full blocks.
    """
    from b210radar.dsp import ToneDownconverter
    rng = np.random.default_rng(0)
    x = (rng.normal(size=200_000) + 1j * rng.normal(size=200_000)).astype(np.complex64)

    ref = ToneDownconverter(cfg)(x)

    dnc, out, i = ToneDownconverter(cfg), [], 0
    while i < x.size:
        k = int(rng.integers(1, 9999))       # deliberately not multiples of 16
        out.append(dnc(x[i:i + k]))
        i += k
    got = np.concatenate([o for o in out if o.shape[1]], axis=1)

    n = min(ref.shape[1], got.shape[1])
    err = float(np.abs(ref[:, :n] - got[:, :n]).max()) if n else np.inf
    return check("downconverter is exact under ragged short reads",
                 n > 0 and ref.shape[1] == got.shape[1] and err < 1e-5,
                 f"max diff {err:.1e} over {n} slow samples")


def main():
    cfg = RadarConfig.sim(clutter_tc=0.5, detect_snr_db=12.0)
    print(cfg.summary())
    print()

    # A hand starting at 1.2 m, closing to 0.4 m at 0.4 m/s, pausing, then
    # withdrawing at 0.6 m/s.
    hand = Target(HAND_RCS, [(0.0, 1.2), (2.0, 0.4), (2.5, 0.4), (3.2, 0.82)])
    # Let the clutter canceller settle and the background tracker warm up
    # before the hand appears.
    settle = max(3 * cfg.clutter_tc,
                 cfg.bg_warmup_frames / cfg.frame_rate) + 1.0
    hand.wp = [(t + settle, r) for t, r in hand.wp]
    duration = settle + 3.4

    dets = run(cfg, [hand], duration)
    print(f"  {len(dets)} frames over {duration:.1f} s\n")

    # Only judge frames whose whole coherent window sees one steady velocity.
    # A frame straddling a velocity change genuinely contains two Doppler
    # tones, so there is no single correct answer to score it against.
    rows = []
    for d in dets:
        # d.time is the centre of the coherent window.
        r_true, v_true = hand.state(d.time)
        _, v_start = hand.state(d.time - cfg.frame_time / 2)
        _, v_end = hand.state(d.time + cfg.frame_time / 2)
        if abs(v_end - v_true) > 1e-6:
            continue
        if abs(v_true) < 0.05 or abs(v_start - v_true) > 1e-6:
            continue
        rows.append((d, r_true, v_true))

    ok = check_streaming(cfg)
    det_rate = np.mean([d.detected for d, _, _ in rows])
    ok &= check("raw per-frame detection while moving > 0.7", det_rate > 0.7,
                f"{det_rate*100:.1f}%  (frames near 1 m sit on the threshold)")

    # What the application actually consumes is the smoothed track, which
    # bridges the marginal frames.
    smoother = TrackSmoother()
    track = []
    for d in dets:
        active, r_s, v_s = smoother.update(d)
        track.append((d, active, r_s, v_s))
    held = [a for d, a, _, _ in track
            if any(d is dd for dd, _, _ in rows)]
    hold_rate = np.mean(held) if held else 0.0
    ok &= check("smoothed track held while moving > 0.95", hold_rate > 0.95,
                f"{hold_rate*100:.1f}%")

    hits = [(d, r, v) for d, r, v in rows if d.detected]

    # Doppler sign: approach_speed > 0 must mean the range is decreasing.
    sign_ok = np.mean([np.sign(d.approach_speed) == np.sign(-v)
                       for d, _, v in hits])
    ok &= check("approach/recede direction correct > 0.95", sign_ok > 0.95,
                f"{sign_ok*100:.1f}%")

    speed_err = np.array([d.approach_speed - (-v) for d, _, v in hits])
    ok &= check("speed error < 3 Doppler bins",
                np.percentile(np.abs(speed_err), 90) < 3 * cfg.speed_resolution,
                f"p90 = {np.percentile(np.abs(speed_err), 90)*100:.1f} cm/s, "
                f"bin = {cfg.speed_resolution*100:.1f} cm/s")

    range_err = np.array([d.range_m - r for d, r, _ in hits])
    med = float(np.median(np.abs(range_err)))
    ok &= check("absolute two-tone range: median error < 0.25 m", med < 0.25,
                f"{med*100:.1f} cm  (static-scene calibration)")
    ok &= check("range: 80% of frames within 0.5 m",
                np.mean(np.abs(range_err) < 0.5) > 0.8,
                f"{np.mean(np.abs(range_err) < 0.5)*100:.1f}%")

    # Referencing the scale to a real echo removes the wall-pull that limits
    # the static-scene calibration.
    from b210radar import phase_offset_for_known_range
    mid = hits[len(hits) // 2]
    better = phase_offset_for_known_range([mid[0]], mid[1], cfg.delta_f)
    corrected = np.array([
        C * np.mod(d.phase_diff - better, 2 * np.pi) / (4 * np.pi * cfg.delta_f) - r
        for d, r, _ in hits])
    med2 = float(np.median(np.abs(corrected)))
    ok &= check("range with known-range calibration: median error < 0.12 m",
                med2 < 0.12, f"{med2*100:.1f} cm")

    # Empty room: nothing moving, so nothing should be detected.
    quiet_cfg = RadarConfig.sim(clutter_tc=0.5, detect_snr_db=12.0)
    quiet_settle = max(3 * quiet_cfg.clutter_tc,
                       quiet_cfg.bg_warmup_frames / quiet_cfg.frame_rate) + 1.0
    quiet = run(quiet_cfg, [], quiet_settle + 4.0)
    tail = [d for d in quiet if d.time > quiet_settle]
    fa = np.mean([d.detected for d in tail]) if tail else 1.0
    ok &= check("false-alarm rate in an empty room < 0.05", fa < 0.05,
                f"{fa*100:.1f}%")

    print()
    print("OK" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
