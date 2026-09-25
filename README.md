# b210_doppler

A two-tone CW Doppler radar for the USRP B210 (and clones such as the LibreSDR B220mini), written in Python. It detects a moving hand, measures how fast it is moving toward or away from the antennas, and gives its absolute range. It includes a live micro-Doppler display, a physics-based simulator that needs no hardware, a few gesture demos (a theremin and mouse scrolling), and a 5.8 GHz patch-antenna PCB you can order.

> Demo project, co-authored with AI.

## Demo

<!-- VIDEO PLACEHOLDER: drag and drop the demo video here on GitHub, or replace with a link -->

https://github.com/user-attachments/assets/REPLACE_ME

## How it works

The B210 transmits two CW tones at the same time, at `f_c + 1.92 MHz` and `f_c + 11.52 MHz` (by default `f_c = 5.8 GHz`, in the ISM band), and receives on a second antenna. TX and RX share one AD9361 and one reference clock, so they stay coherent.

```
USRP RX ──► tone downconverter ──► clutter canceller ──► Doppler FFT ──► CFAR detect ──► speed + range
            (mix + CIC + FIR,      (slow IIR high-pass,  (512-pt Hann,   (temporal median
             decim 16384)           removes leakage,      128 hop)        + GO-CFAR floor)
                                    walls, furniture)
```

- **Speed** comes from the Doppler shift of the strongest moving return. Positive speed means the target is getting closer.
- **Range** comes from the phase difference between the two tones at the Doppler peak: `R = c · Δφ / (4π · Δf)`. With `Δf = 9.6 MHz` the range is unambiguous out to about 15.6 m.
- **Static clutter** (TX→RX leakage, walls) is tracked and subtracted. Its two-tone phase also gives the hardware delay needed to calibrate range zero.
- **Detection** compares each Doppler bin with a per-bin running median over time, floored by a greatest-of CFAR. This handles the steep phase-noise pedestal around zero Doppler, which a plain CFAR does not.

Default hardware profile:

| Parameter              | Value                                  |
|------------------------|----------------------------------------|
| Carrier                | 5.80 GHz (λ ≈ 5.16 cm)                 |
| Tones                  | +1.92 / +11.52 MHz (Δf = 9.6 MHz)      |
| Sample rate            | 30.72 MSps → 1875 Hz slow-time         |
| Unambiguous range      | 15.6 m                                 |
| Max radial speed       | ±24.2 m/s                              |
| Speed resolution       | 9.5 cm/s (273 ms coherent integration) |
| Frame rate             | 14.6 Hz                                |

### What limits range

The limit is TX→RX antenna isolation, not transmit power. LO phase noise mixes with the direct leakage and creates a noise pedestal around zero Doppler, and that pedestal goes up or down dB for dB with the leakage. Use two separate antennas, keep them far apart, and cross-polarise them if you can. `scripts/range_sweep.py` shows the effect in simulation.

If you have a directional coupler, `--ref-channel` feeds a TX reference into RX channel 1 and cancels the LO phase noise.

## Repository layout

```
b210radar/            core library
  config.py           RadarConfig: RF/DSP parameters, hardware and sim presets
  dsp.py              tone downconverter, FIR decimators, clutter canceller
  detect.py           RadarProcessor, detection, calibration, track smoothing
  source.py           UsrpSource (real B210) and SimSource (simulator)
  display.py          live micro-Doppler waterfall + range plot (matplotlib)
scripts/
  radar.py            live hand detection with display
  theremin.py         range → pitch, speed → vibrato, presence → volume
  scroll.py           hand motion → mouse wheel scrolling
  hw_check.py         hardware check: confirms both tones arrive TX → RX
  selftest.py         end-to-end validation against a simulated hand
  range_sweep.py      detection range vs. target RCS and antenna isolation
hardware/
  patch_2x2_5g8/      2x2 inset-fed microstrip patch array, 5.8 GHz, JLC FR-4
```

## Installation

```sh
python3 -m venv .venv
source .venv/bin/activate        # fish: source .venv/bin/activate.fish
pip install -r requirements.txt
```

The UHD Python bindings are only needed for real hardware. Install them system-wide with libuhd (for example `pacman -S libuhd` or `apt install python3-uhd`). If you use a venv, create it with `--system-site-packages` so it can see them.

## Usage

### Without hardware

```sh
python3 scripts/selftest.py            # checks detection, Doppler sign, speed and range accuracy
python3 scripts/radar.py --sim         # live display with a simulated hand
python3 scripts/range_sweep.py         # how far can it see a hand / head / person?
```

### With a B210

Connect one antenna to **TX/RX** and one to **RX2**, then:

```sh
python3 scripts/hw_check.py            # both tones should show "OK"
python3 scripts/radar.py --usrp        # live display
```

When it starts, the radar spends 3 s calibrating range zero on the static scene, so keep the area in front of the antennas still. For better range accuracy, also reference the range scale to a real echo: keep a hand moving at a measured distance while it calibrates.

```sh
python3 scripts/radar.py --usrp --calibrate-at 0.5
```

Useful options (shared by `radar.py`, `theremin.py` and `scroll.py`):

| Option              | Meaning                                                  |
|---------------------|----------------------------------------------------------|
| `--freq HZ`         | carrier frequency (default `5.8e9`)                      |
| `--tx-gain DB`      | TX gain, 0–89.75 (default 70)                            |
| `--rx-gain DB`      | RX gain, 0–76 (default 60)                               |
| `--snr DB`          | detection threshold above background (default 12)        |
| `--min-speed M/S`   | slower movers are treated as clutter (default 0.10)      |
| `--ref-channel`     | RX channel 1 is a coupler reference (cancels phase noise)|
| `--calibrate SEC`   | seconds of empty-scene calibration (default 3)           |
| `--calibrate-at M`  | extra calibration against a mover at a known distance    |
| `--device-args STR` | UHD device args                                          |
| `--leakage-db DB`   | simulator only: TX→RX isolation                          |

`radar.py` also takes `--duration SEC` and `--no-display` (text output only).

### Demos

```sh
# Theremin: range sets pitch, speed adds vibrato
python3 scripts/theremin.py --usrp --min-freq 200 --max-freq 1200 --snap --waveform triangle

# Scroll the mouse wheel by moving a hand toward / away from the antennas
python3 scripts/scroll.py --usrp --scroll-speed normal [--invert] [--sensitivity 1.5]
```

Both also accept `--sim`. In simulator mode, `scroll.py` prints the scroll events instead of sending them.

## Antenna hardware

`hardware/patch_2x2_5g8/` contains a 2x2 inset-fed microstrip patch array for 5.8 GHz. It is 100 x 100 mm, 2-layer, 1.6 mm FR-4, and has an edge SMA. Order one design and use **two boards**, one on TX/RX and one on RX2. Two separate boards give much better isolation than anything that fits on one board.

- `patch_2x2_5g8_jlc.zip`: Gerbers and drill files ready to upload to JLCPCB
- `patch_2x2_5g8.kicad_pcb`: KiCad board
- `gen_patch_array.py`: generates the board with KiCad's `pcbnew` Python API. Change `PATCH_L` to retune it

FR-4 permittivity varies at 6 GHz, so the resonance may land slightly off. Measure it with a VNA and trim `PATCH_L` if you need to.

## Safety and regulations

Transmit only in a band you are allowed to use, and at low power. The defaults stay inside the 5.725–5.875 GHz ISM band at a few milliwatts. Check your local rules before transmitting.

## License

[GNU AGPL-3.0](LICENSE)
