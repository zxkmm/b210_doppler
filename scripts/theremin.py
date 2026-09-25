#!/usr/bin/env python3
"""Theremin controlled by b210radar Doppler radar.

Range controls pitch, speed controls vibrato, presence controls volume.
"""

import argparse
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from b210radar import (RadarConfig, RadarProcessor, SimSource,
                       Target, TrackSmoother, with_overrides,
                       phase_offset_for_known_range)

try:
    import sounddevice as sd
except ImportError:
    sd = None

# Note names for the chromatic scale display
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def freq_to_note_name(freq):
    if freq <= 0:
        return "--"
    # A4 = 440Hz
    midi_note = int(round(69 + 12 * math.log2(freq / 440.0)))
    name = NOTE_NAMES[midi_note % 12]
    octave = (midi_note // 12) - 1
    return f"{name}{octave}"

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
                   help="RX2 channel 1 carries a coupler reference")
    p.add_argument("--leakage-db", type=float, dest="sim_leakage_db",
                   help="simulator: TX->RX antenna isolation in dB")
    p.add_argument("--calibrate", type=float, default=3.0, metavar="SEC",
                   help="seconds of empty-scene calibration for range zero")
    p.add_argument("--calibrate-at", type=float, metavar="METRES",
                   help="calibrate to a known target distance")

    # Theremin specific args
    p.add_argument("--min-freq", type=float, default=200.0, help="lowest frequency (Hz)")
    p.add_argument("--max-freq", type=float, default=1200.0, help="highest frequency (Hz)")
    p.add_argument("--snap", action="store_true", help="enable chromatic scale snapping")
    p.add_argument("--volume", type=float, default=0.5, help="master volume 0.0-1.0")
    p.add_argument("--waveform", choices=['sine', 'triangle', 'saw'], default='sine')
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

    script, t = [], 4.0
    r = 1.5
    for _ in range(20):
        script += [(t, r), (t + 2.5, 0.3), (t + 3.0, 0.3), (t + 4.5, r)]
        t += 6.0
    return SimSource(cfg, targets=[Target(0.01, script)], realtime=True)

def calibrate(src, proc, cfg, seconds):
    if seconds <= 0:
        return
    print(f"calibrating range zero on the static scene for {seconds:.1f} s ...", flush=True)
    n = int(seconds * cfg.samp_rate / cfg.fast_block)
    for _ in range(n):
        list(proc.process(src.recv()))
    cfg.range_phase_offset = proc.leakage_phase()
    print(f"  range phase offset = {cfg.range_phase_offset:+.4f} rad", flush=True)

def calibrate_known(src, cfg, true_range, seconds=6.0):
    print(f"move a hand steadily at {true_range:.2f} m for {seconds:.0f} s...", flush=True)
    proc = RadarProcessor(cfg)
    dets = []
    for _ in range(int(seconds * cfg.samp_rate / cfg.fast_block)):
        dets.extend(proc.process(src.recv()))
    usable = [d for d in dets if d.detected]
    if not usable:
        print("  no mover seen -- keeping the static-scene calibration")
        return
    cfg.range_phase_offset = phase_offset_for_known_range(usable, true_range, cfg.delta_f)
    print(f"  {len(usable)} detections -> offset {cfg.range_phase_offset:+.4f} rad", flush=True)


class ThereminState:
    def __init__(self):
        self.active = False
        self.distance = 1.0
        self.speed = 0.0

class AudioSynth:
    def __init__(self, sample_rate, state, args):
        self.sample_rate = sample_rate
        self.state = state
        self.min_freq = args.min_freq
        self.max_freq = args.max_freq
        self.snap = args.snap
        self.master_vol = args.volume
        self.waveform = args.waveform
        
        self.phase = 0.0
        self.current_vol = 0.0
        
        # Audio rendering params
        self.vol_step_up = self.master_vol / (0.100 * self.sample_rate)  # 100ms up
        self.vol_step_down = self.master_vol / (0.200 * self.sample_rate) # 200ms down
        self.vib_phase = 0.0
        self.vib_rate = 6.0 # Hz
        
        # Info for UI
        self.ui_freq = 0.0
        self.ui_vol = 0.0

    def calc_freq(self, r):
        r_min, r_max = 0.2, 2.0
        r_clamped = max(r_min, min(r_max, r))
        # Logarithmic mapping: closer = higher freq
        ratio = (r_max - r_clamped) / (r_max - r_min)
        f = self.min_freq * (self.max_freq / self.min_freq) ** ratio
        if self.snap:
            # Snap to nearest chromatic note
            f = 440.0 * (2.0 ** (round(12.0 * math.log2(f / 440.0)) / 12.0))
        return f

    def audio_callback(self, outdata, frames, time_info, status):
        # Read atomic state (GIL makes float reads thread-safe)
        active = self.state.active
        r = self.state.distance
        speed = abs(self.state.speed)
        
        target_vol = self.master_vol if active else 0.0
        base_freq = self.calc_freq(r) if active else self.ui_freq # maintain freq if fading out
        
        vib_amp = speed * 30.0
        
        # Generate samples
        for i in range(frames):
            # Volume envelope
            if self.current_vol < target_vol:
                self.current_vol = min(target_vol, self.current_vol + self.vol_step_up)
            elif self.current_vol > target_vol:
                self.current_vol = max(target_vol, self.current_vol - self.vol_step_down)
                
            # Vibrato
            vib_freq = base_freq + vib_amp * math.sin(2.0 * math.pi * self.vib_phase)
            self.vib_phase = (self.vib_phase + self.vib_rate / self.sample_rate) % 1.0
            
            # Phase update
            self.phase = (self.phase + vib_freq / self.sample_rate) % 1.0
            
            # Waveform gen
            if self.waveform == 'sine':
                val = math.sin(2.0 * math.pi * self.phase)
            elif self.waveform == 'triangle':
                val = 4.0 * abs(self.phase - 0.5) - 1.0
            else: # saw
                val = 2.0 * self.phase - 1.0
                
            outdata[i, 0] = val * self.current_vol
            
            if i == frames - 1:
                self.ui_freq = base_freq
                self.ui_vol = self.current_vol


def print_ui(synth, min_f, max_f):
    freq = synth.ui_freq
    vol = synth.ui_vol
    
    # Pitch bar
    bar_len = 40
    # Map frequency logarithmically to bar position
    if freq > 0 and min_f > 0:
        ratio = math.log2(freq / min_f) / math.log2(max_f / min_f)
    else:
        ratio = 0.0
    pos = int(max(0, min(bar_len - 1, ratio * bar_len)))
    pitch_bar = ["-"] * bar_len
    if vol > 0.01:
        pitch_bar[pos] = "O"
    else:
        pitch_bar = [" "] * bar_len
    
    # Vol bar
    vol_len = 10
    v_pos = int((vol / max(0.01, synth.master_vol)) * vol_len)
    v_pos = max(0, min(vol_len, v_pos))
    vol_bar = "#" * v_pos + " " * (vol_len - v_pos)
    
    note_name = freq_to_note_name(freq) if vol > 0.01 else "--"
    freq_str = f"{freq:6.1f} Hz" if vol > 0.01 else "   --    "
    
    print(f"\rPitch: [{''.join(pitch_bar)}] {freq_str} ({note_name:3s}) | Vol: [{vol_bar}]", end="", flush=True)


BANNER = r"""
  _____ _                          _
 |_   _| |__   ___ _ __ ___ _ __ (_)_ __
   | | | '_ \ / _ \ '__/ _ \ '_ \| | '_ \
   | | | | | |  __/ | |  __/ | | | | | | |
   |_| |_| |_|\___|_|  \___|_| |_|_|_| |_|
    ~ Doppler Radar Theremin ~
"""


def main(argv=None):
    a = build_args().parse_args(argv)

    if sd is None:
        print("Error: sounddevice module not found. Please pip install sounddevice.")
        return 1

    cfg = make_config(a)
    print(cfg.summary())
    
    src = make_source(a, cfg)
    proc = RadarProcessor(cfg)
    src.start()
    
    try:
        calibrate(src, proc, cfg, a.calibrate)
        if a.calibrate_at:
            calibrate_known(src, cfg, a.calibrate_at)
        proc = RadarProcessor(cfg)

        state = ThereminState()
        synth = AudioSynth(44100, state, a)
        
        stop = threading.Event()
        
        def capture():
            smoother = TrackSmoother()
            while not stop.is_set():
                for det in proc.process(src.recv()):
                    active, r_s, v_s = smoother.update(det)
                    state.active = active
                    state.distance = r_s
                    state.speed = v_s

        worker = threading.Thread(target=capture, daemon=True)
        worker.start()
        
        print(BANNER)
        print("Theremin active -- Ctrl-C to stop\n", flush=True)

        stream = sd.OutputStream(
            samplerate=44100,
            blocksize=1024,
            channels=1,
            callback=synth.audio_callback
        )
        
        with stream:
            while worker.is_alive():
                print_ui(synth, a.min_freq, a.max_freq)
                time.sleep(0.125) # ~8 Hz UI update
                
    except KeyboardInterrupt:
        print("\n\ninterrupted")
    finally:
        stop.set()
        if 'worker' in locals():
            worker.join(timeout=3.0)
        src.stop()
        if src.status:
            print(f"stream issues: {src.status}")
            
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
