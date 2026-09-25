#!/usr/bin/env python3
"""Generate a 2x2 inset-fed microstrip patch array for 5.8 GHz on JLC FR-4.

Order one design and use two copies: one on TX/RX, one on RX2. Physical
separation between two boards buys far more TX->RX isolation than anything
that fits on a single 100 x 100 mm board, and isolation is what limits this
radar (see b210radar/config.py, sim_leakage_db).

Stack-up: JLC 2-layer, 1.6 mm FR-4, 1 oz copper. F.Cu = antenna, B.Cu = ground.
Coordinates below are mm relative to the board centre, y pointing down
(towards the SMA). Run with KiCad's Python:  python3 gen_patch_array.py
"""

import math
from pathlib import Path

import pcbnew

# ---- design ---------------------------------------------------------------
C = 299_792_458.0
F0 = 5.8067e9          # centre of the two radiated tones at the default carrier
ER = 4.3               # FR-4 at ~6 GHz; JLC quote 4.5-4.6 at 1 GHz
H = 1.51               # dielectric thickness of a 1.6 mm board with 1 oz Cu

BOARD = 100.0          # square, the JLC "free" size limit

PATCH_W = 15.86        # non-radiating edges (along x)
PATCH_L = 11.91        # resonant length (along y) -- trim this to tune up
INSET = 3.68           # inset depth giving 100 ohm at the notch tip
NOTCH_GAP = 0.80       # copper gap either side of the feed in the notch

W100 = 0.69            # 100 ohm microstrip
W50 = 2.94             # 50 ohm microstrip
W35 = 5.00             # 35.4 ohm quarter-wave transformer, 25 -> 50 ohm
QW35 = 6.98            # its length (lambda_g / 4)
LG50 = 28.57           # one guided wavelength on the 50 ohm trunk

DX = 36.0              # element pitch across (0.70 lambda0)
DY = LG50              # element pitch along = trunk length, keeps rows in phase
FEED_CLEAR = 4.0       # distance from a patch's fed edge to its 100 ohm T line

# ---- board furniture --------------------------------------------------------
ORIGIN = (100.0, 100.0)   # page position of the board centre
EDGE_PULLBACK = 0.3
MOUNT_INSET = 4.0
FP_LIB = "/usr/share/kicad/footprints"

HERE = Path(__file__).resolve().parent
OUT = HERE / "patch_2x2_5g8.kicad_pcb"


def mm(x, y):
    return pcbnew.VECTOR2I(pcbnew.FromMM(ORIGIN[0] + x), pcbnew.FromMM(ORIGIN[1] + y))


def poly(board, pts, layer, net):
    s = pcbnew.PCB_SHAPE(board)
    s.SetShape(pcbnew.SHAPE_T_POLY)
    s.SetPolyPoints([mm(*p) for p in pts])
    s.SetLayer(layer)
    s.SetFilled(True)
    s.SetWidth(0)
    if net is not None:
        s.SetNet(net)
    board.Add(s)


def rect(board, x0, y0, x1, y1, layer, net):
    poly(board, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], layer, net)


def hline(board, x0, x1, y, w, net):
    rect(board, min(x0, x1) - w / 2, y - w / 2, max(x0, x1) + w / 2, y + w / 2, pcbnew.F_Cu, net)


def vline(board, x, y0, y1, w, net):
    rect(board, x - w / 2, min(y0, y1), x + w / 2, max(y0, y1), pcbnew.F_Cu, net)


def patch(board, cx, cy, net):
    """Patch fed on its lower (+y) edge through an inset notch."""
    x0, x1 = cx - PATCH_W / 2, cx + PATCH_W / 2
    y0, y1 = cy - PATCH_L / 2, cy + PATCH_L / 2
    n = W100 / 2 + NOTCH_GAP
    pts = [(x0, y0), (x1, y0), (x1, y1),
           (cx + n, y1), (cx + n, y1 - INSET), (cx - n, y1 - INSET), (cx - n, y1),
           (x0, y1)]
    poly(board, pts, pcbnew.F_Cu, net)
    tip = y1 - INSET
    return tip, y1


def load_fp(board, lib, name, ref, x, y, rot=0.0):
    fp = pcbnew.FootprintLoad(f"{FP_LIB}/{lib}.pretty", name)
    fp.SetParent(board)
    fp.SetReference(ref)
    fp.SetPosition(mm(x, y))
    fp.SetOrientationDegrees(rot)
    board.Add(fp)
    return fp


def text(board, s, x, y, size=1.2, layer=pcbnew.F_SilkS):
    t = pcbnew.PCB_TEXT(board)
    t.SetText(s)
    t.SetPosition(mm(x, y))
    t.SetLayer(layer)
    t.SetTextSize(pcbnew.VECTOR2I(pcbnew.FromMM(size), pcbnew.FromMM(size)))
    t.SetTextThickness(pcbnew.FromMM(size * 0.15))
    if layer == pcbnew.B_SilkS:
        t.SetMirrored(True)
    board.Add(t)


def main():
    board = pcbnew.CreateEmptyBoard()
    board.SetCopperLayerCount(2)
    ds = board.GetDesignSettings()
    ds.SetBoardThickness(pcbnew.FromMM(1.6))
    ds.m_CopperEdgeClearance = pcbnew.FromMM(0.2)

    rf = pcbnew.NETINFO_ITEM(board, "RF")
    gnd = pcbnew.NETINFO_ITEM(board, "GND")
    board.Add(rf)
    board.Add(gnd)

    half = BOARD / 2
    edge = pcbnew.PCB_SHAPE(board)
    edge.SetShape(pcbnew.SHAPE_T_RECT)
    edge.SetStart(mm(-half, -half))
    edge.SetEnd(mm(half, half))
    edge.SetLayer(pcbnew.Edge_Cuts)
    edge.SetWidth(pcbnew.FromMM(0.1))
    board.Add(edge)

    # ---- array ----
    # Both rows are fed from their lower edge, so they radiate in phase when
    # the trunk between the two T junctions is exactly one guided wavelength.
    ys = (-DY / 2, DY / 2)
    t_ys = []
    for cy in ys:
        t_y = cy + PATCH_L / 2 + FEED_CLEAR
        for cx in (-DX / 2, DX / 2):
            tip, _ = patch(board, cx, cy, rf)
            vline(board, cx, tip, t_y + W100 / 2, W100, rf)
        hline(board, -DX / 2, DX / 2, t_y, W100, rf)
        t_ys.append(t_y)

    # Trunk, top T -> node. At the node: 100 || 100 || 50 = 25 ohm.
    node = t_ys[1]
    assert abs((node - t_ys[0]) - LG50) < 1e-6
    vline(board, 0, t_ys[0] - W50 / 2, node, W50, rf)

    # 25 -> 50 ohm quarter-wave, then 50 ohm to the edge SMA.
    qw_end = node + QW35
    vline(board, 0, node, qw_end, W35, rf)
    sma_y = half - 2.54            # footprint origin sits 2.54 mm inside the edge
    vline(board, 0, qw_end, sma_y, W50, rf)

    j1 = load_fp(board, "Connector_Coaxial", "SMA_Amphenol_132289_EdgeMount",
                 "J1", 0, sma_y, 270)
    for p in j1.Pads():
        p.SetNet(rf if p.GetNumber() == "1" else gnd)

    # Grounded coplanar shoulders on the top side for the connector launch.
    for s in (-1, 1):
        x0, x1 = s * 3.5, s * 9.5
        rect(board, min(x0, x1), half - 9.0, max(x0, x1), half - EDGE_PULLBACK,
             pcbnew.F_Cu, gnd)
        for vx in (5.8, 7.9):
            for vy in (half - 7.8, half - 5.8, half - 3.8, half - 1.8):
                v = pcbnew.PCB_VIA(board)
                v.SetPosition(mm(s * vx, vy))
                v.SetDrill(pcbnew.FromMM(0.4))
                v.SetWidth(pcbnew.FromMM(0.8))
                v.SetNet(gnd)
                board.Add(v)

    # ---- ground plane ----
    z = pcbnew.ZONE(board)
    z.SetLayer(pcbnew.B_Cu)
    z.SetNet(gnd)
    z.SetIsFilled(False)
    z.SetLocalClearance(pcbnew.FromMM(0.3))
    z.SetMinThickness(pcbnew.FromMM(0.25))
    z.SetPadConnection(pcbnew.ZONE_CONNECTION_FULL)
    ol = z.Outline()
    ol.NewOutline()
    g = half - EDGE_PULLBACK
    for x, y in ((-g, -g), (g, -g), (g, g), (-g, g)):
        ol.Append(pcbnew.FromMM(ORIGIN[0] + x), pcbnew.FromMM(ORIGIN[1] + y))
    board.Add(z)

    # ---- mounting ----
    m = half - MOUNT_INSET
    for i, (x, y) in enumerate(((-m, -m), (m, -m), (-m, m), (m, m)), 1):
        load_fp(board, "MountingHole", "MountingHole_3.2mm_M3", f"H{i}", x, y)

    # ---- silk ----
    text(board, "5.8 GHz 2x2 patch", 0, -half + 4.0, 1.5)
    text(board, "b210_doppler  TX / RX", 0, -half + 6.5, 1.0)
    text(board, "E-plane", -half + 12, 0, 1.0)
    text(board, "5.8 GHz 2x2 patch array  JLC FR-4 1.6 mm", 0, 0, 1.5, pcbnew.B_SilkS)
    text(board, f"W {PATCH_W}  L {PATCH_L}  inset {INSET}  er {ER}", 0, 3, 1.0, pcbnew.B_SilkS)
    text(board, "trim L (radiating edges) to raise f", 0, 5.5, 1.0, pcbnew.B_SilkS)
    # E-plane arrow: polarisation is along y
    for a, b in (((-half + 8, -6), (-half + 8, 6)),
                 ((-half + 8, -6), (-half + 7, -4.5)), ((-half + 8, -6), (-half + 9, -4.5)),
                 ((-half + 8, 6), (-half + 7, 4.5)), ((-half + 8, 6), (-half + 9, 4.5))):
        s = pcbnew.PCB_SHAPE(board)
        s.SetShape(pcbnew.SHAPE_T_SEGMENT)
        s.SetStart(mm(*a))
        s.SetEnd(mm(*b))
        s.SetLayer(pcbnew.F_SilkS)
        s.SetWidth(pcbnew.FromMM(0.2))
        board.Add(s)

    pcbnew.ZONE_FILLER(board).Fill(board.Zones())
    board.Save(str(OUT))
    print(f"wrote {OUT}")
    print(f"lambda0 {C / F0 * 1e3:.2f} mm, pitch {DX / (C / F0 * 1e3):.2f} x "
          f"{DY / (C / F0 * 1e3):.2f} lambda0, er {ER}, h {H} mm")


if __name__ == "__main__":
    main()
