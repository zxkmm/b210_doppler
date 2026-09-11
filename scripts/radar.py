#!/usr/bin/env python3
"""Live hand detection with a USRP B210 (or the built-in simulator).

    # simulated hand, no hardware needed
    python3 scripts/radar.py --sim

    # real B210, two antennas: TX/RX transmits, RX2 receives
    python3 scripts/radar.py --usrp --freq 5.8e9

Transmit only into a band you are allowed to use, at low power, and keep the
antennas pointed away from people's eyes at close range.  The defaults sit in
the 5.725-5.875 GHz ISM band at a few milliwatts.
"""

import argparse
import threading
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from b210radar import (RadarConfig, RadarProcessor, SimSource,  # noqa: E402
                       Target, TrackSmoother, with_overrides,
                       phase_offset_for_known_range)


def build_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--usrp", action="store_true", help="use a real B210")
    src.add_argument("--sim", action="store_true", help="use the simulator")

    p.add_argument("--device-args", default="", help="UHD device args")
    p.add_argument("--freq", type=float, help="carrier in Hz (default 5.8e9)")
    p.add_argument("--tx-gain", type=float, help="dB, 0..89.75")
    p.add_argument("--rx-gain", type=float, help="dB, 0..76")
    p.add_argument("--min-speed", type=float, help="m/s detection floor")
    p.add_argument("--snr", type=float, dest="detect_snr_db",
                   help="detection threshold in dB over background")
    p.add_argument("--ref-channel", action="store_true",
                   help="RX2 channel 1 carries a coupler reference; cancels LO "
                        "phase noise (needs a directional coupler on the TX line)")
    p.add_argument("--leakage-db", type=float, dest="sim_leakage_db",
                   help="simulator: TX->RX antenna isolation in dB")
    p.add_argument("--duration", type=float, default=0.0,
                   help="seconds to run (0 = until interrupted)")
    p.add_argument("--no-display", action="store_true", help="text output only")
    p.add_argument("--calibrate", type=float, default=3.0, metavar="SEC",
                   help="seconds of empty-scene calibration for range zero")
    p.add_argument("--calibrate-at", type=float, metavar="METRES",
                   help="after the empty-scene pass, keep a mover at this "
                        "measured distance for a few seconds to reference the "
                        "range scale to a real echo (much more accurate)")
    return p


def make_config(a):
    cfg = RadarConfig.sim() if a.sim else RadarConfig.hardware()
    cfg = with_overrides(
        cfg,
        center_freq=a.freq, tx_gain=a.tx_gain, rx_gain=a.rx_gain,
        min_speed=a.min_speed, detect_snr_db=a.detect_snr_db,
        sim_leakage_db=a.sim_leakage_db,
        rx_channels=(0, 1) if a.ref_channel else None,
    )
    cfg.validate()
    return cfg


def make_source(a, cfg):
    if a.usrp:
        from b210radar import UsrpSource
        return UsrpSource(cfg, device_args=a.device_args)

    # A scripted hand: waits, closes from 1.5 m to 0.3 m, holds, withdraws,
    # then repeats. Ranges are absolute time in seconds.
    script, t = [], 4.0
    r = 1.5
    for _ in range(20):
        script += [(t, r), (t + 2.5, 0.3), (t + 3.0, 0.3), (t + 4.5, r)]
        t += 6.0
    return SimSource(cfg, targets=[Target(0.01, script)], realtime=True)


def calibrate(src, proc, cfg, seconds):
    """Zero the range scale on the TX->RX leakage path."""
    if seconds <= 0:
        return
    print(f"calibrating range zero on the static scene for {seconds:.1f} s "
          f"-- keep the field of view clear...", flush=True)
    n = int(seconds * cfg.samp_rate / cfg.fast_block)
    for _ in range(n):
        list(proc.process(src.recv()))
    cfg.range_phase_offset = proc.leakage_phase()
    print(f"  range phase offset = {cfg.range_phase_offset:+.4f} rad "
          f"({cfg.range_phase_offset * 299792458.0 / (4*np.pi*cfg.delta_f):+.3f} m)",
          flush=True)


def calibrate_known(src, cfg, true_range, seconds=6.0):
    """Reference the range scale to a mover at a measured distance."""
    print(f"\nmove a hand steadily at {true_range:.2f} m for {seconds:.0f} s...",
          flush=True)
    proc = RadarProcessor(cfg)
    dets = []
    for _ in range(int(seconds * cfg.samp_rate / cfg.fast_block)):
        dets.extend(proc.process(src.recv()))
    usable = [d for d in dets if d.detected]
    if not usable:
        print("  no mover seen -- keeping the static-scene calibration")
        return
    cfg.range_phase_offset = phase_offset_for_known_range(
        usable, true_range, cfg.delta_f)
    print(f"  {len(usable)} detections -> range phase offset "
          f"{cfg.range_phase_offset:+.4f} rad", flush=True)


def main(argv=None):
    a = build_args().parse_args(argv)
    cfg = make_config(a)
    print(cfg.summary())
    print()

    src = make_source(a, cfg)
    proc = RadarProcessor(cfg)
    src.start()
    try:
        calibrate(src, proc, cfg, a.calibrate)
        if a.calibrate_at:
            calibrate_known(src, cfg, a.calibrate_at)
        # Rebuild so the freshly calibrated offset is used, then let the
        # background tracker warm up again.
        proc = RadarProcessor(cfg)

        disp = None
        if not a.no_display:
            try:
                from b210radar.display import LiveDisplay
                disp = LiveDisplay(cfg)
            except Exception as exc:                     # noqa: BLE001
                print(f"display unavailable ({exc}); falling back to text")

        stop = threading.Event()
        print("running -- Ctrl-C to stop\n", flush=True)

        def capture():
            """Drain the SDR and process it, never touching the GUI.

            Detection and tracking live here so that a slow or stalled
            renderer cannot drop samples or break the track's hysteresis.
            """
            smoother = TrackSmoother()
            last_print = 0.0
            while not stop.is_set():
                for det in proc.process(src.recv()):
                    active, r_s, v_s = smoother.update(det)
                    if disp is not None:
                        disp.push(det, active, r_s, v_s, src.status)
                    elif det.time - last_print > 0.25:
                        last_print = det.time
                        if active:
                            print(f"  t={det.time:7.2f}s  "
                                  f"{'CLOSING' if v_s > 0 else 'RECEDING':8s}  "
                                  f"range {r_s:5.2f} m  speed {abs(v_s):4.2f} m/s  "
                                  f"SNR {det.snr_db:5.1f} dB", flush=True)
                        else:
                            print(f"  t={det.time:7.2f}s  --", flush=True)

        worker = threading.Thread(target=capture, daemon=True)
        worker.start()
        t0 = time.monotonic()
        try:
            while worker.is_alive():
                if a.duration and time.monotonic() - t0 > a.duration:
                    break
                if disp is not None:
                    if not disp.alive:
                        break
                    disp.render()
                    time.sleep(0.02)
                else:
                    time.sleep(0.1)
        finally:
            stop.set()
            worker.join(timeout=3.0)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        src.stop()
        if src.status:
            print(f"stream issues: {src.status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
