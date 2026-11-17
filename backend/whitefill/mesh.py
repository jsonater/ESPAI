"""Write a watertight binary STL (millimetres) from an outline mask and a height."""

from __future__ import annotations

import struct

import numpy as np


def masked_heightmap_to_stl(hm, inc, cell_mm, path):
    """Solid = cells where inc is True; top follows hm, flat bottom at z=0, walls on the outline."""
    R, C = inc.shape
    hz = np.where(inc, hm, 0.0)
    pad = np.pad(hz, 1); cnt = np.pad(inc.astype(float), 1)
    s = pad[:-1, :-1] + pad[:-1, 1:] + pad[1:, :-1] + pad[1:, 1:]
    n = cnt[:-1, :-1] + cnt[:-1, 1:] + cnt[1:, :-1] + cnt[1:, 1:]
    Z = s / np.maximum(n, 1)                            # corner heights (R+1, C+1)
    X, Y = np.meshgrid(np.arange(C + 1) * cell_mm, (R - np.arange(R + 1)) * cell_mm)
    top = np.stack([X, Y, Z], -1); bot = np.stack([X, Y, np.zeros_like(Z)], -1)

    rr, cc = np.nonzero(inc)
    TL, TR = (rr, cc), (rr, cc + 1)
    BR, BL = (rr + 1, cc + 1), (rr + 1, cc)
    quads = [np.stack([top[BL], top[BR], top[TR], top[TL]], 1),         # up
             np.stack([bot[TL], bot[TR], bot[BR], bot[BL]], 1)]         # down
    pin = np.pad(inc, 1)
    def edge(sel, a, b, c, d):
        if sel.any():
            quads.append(np.stack([a[sel], b[sel], c[sel], d[sel]], 1))
    up, dn = ~pin[rr, cc + 1], ~pin[rr + 2, cc + 1]
    lf, rt = ~pin[rr + 1, cc], ~pin[rr + 1, cc + 2]
    g = lambda arr, ix: arr[ix]
    edge(up, g(bot, TR), g(bot, TL), g(top, TL), g(top, TR))
    edge(dn, g(bot, BL), g(bot, BR), g(top, BR), g(top, BL))
    edge(lf, g(bot, TL), g(bot, BL), g(top, BL), g(top, TL))
    edge(rt, g(bot, BR), g(bot, TR), g(top, TR), g(top, BR))
    q = np.concatenate(quads).astype(np.float32)
    tris = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    with open(path, "wb") as f:
        f.write(b"ESPAI whitefill part (mm)".ljust(80, b" "))
        f.write(struct.pack("<I", len(tris)))
        rec = np.zeros(len(tris), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        rec["n"], rec["v"] = nrm, tris
        f.write(rec.tobytes())
    return len(tris)


def prism_stl(inc: np.ndarray, cell_mm: float, height_mm: float, path: str) -> int:
    """Solid of even height over the True cells of `inc`. Returns the triangle count."""
    return masked_heightmap_to_stl(np.full(inc.shape, float(height_mm), np.float32), inc, cell_mm, path)
