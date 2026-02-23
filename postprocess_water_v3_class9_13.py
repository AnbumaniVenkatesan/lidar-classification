import os
import numpy as np
import laspy
from tqdm import tqdm
from scipy.ndimage import label

IN_LAZ  = r"d:\lidarrrrr\anbu\classified_output_HAGfix_out7.laz"
OUT_LAZ = r"d:\lidarrrrr\anbu\classified_output_WATERFIX_v3.laz"

SEA_CLASS  = 9
LAKE_CLASS = 13

# ----- Tune here -----
GRID = 2.0                  # ✅ coarser grid connects sea better (try 2.0 or 5.0)
MIN_POINTS_PER_CELL = 3

Z_STD_MAX   = 0.18          # meters (try 0.15 ~ 0.25)
Z_RANGE_MAX = 0.50          # meters (try 0.40 ~ 0.80)

USE_INTENSITY = True
INTENSITY_MAX = 0.45        # normalized [0..1] (raise if too strict)

MIN_INLAND_M = 10.0         # inland water components must be >= 10m extent
# ----------------------

def main():
    if not os.path.exists(IN_LAZ):
        raise FileNotFoundError(IN_LAZ)

    print("Reading:", IN_LAZ)
    las = laspy.read(IN_LAZ)

    x = np.array(las.x, dtype=np.float32)
    y = np.array(las.y, dtype=np.float32)
    z = np.array(las.z, dtype=np.float32)
    cls = np.array(las.classification, dtype=np.int32)

    dims = set(las.point_format.dimension_names)
    has_intensity = "intensity" in dims
    print("Intensity available:", has_intensity)

    if USE_INTENSITY and has_intensity:
        intensity = np.array(las.intensity, dtype=np.float32)
        intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-6)
    else:
        intensity = None

    xmin, ymin = x.min(), y.min()
    ix = np.floor((x - xmin) / GRID).astype(np.int32)
    iy = np.floor((y - ymin) / GRID).astype(np.int32)
    nx = int(ix.max()) + 1
    ny = int(iy.max()) + 1

    # Welford per-cell mean/std + min/max
    cnt  = np.zeros((nx, ny), dtype=np.int32)
    mean = np.zeros((nx, ny), dtype=np.float32)
    m2   = np.zeros((nx, ny), dtype=np.float32)
    zmin = np.full((nx, ny), np.inf, dtype=np.float32)
    zmax = np.full((nx, ny), -np.inf, dtype=np.float32)

    sumi = np.zeros((nx, ny), dtype=np.float32) if intensity is not None else None

    print("Building grid stats (mean/std/range)...")
    for i in tqdm(range(len(x)), desc="Grid stats"):
        cx, cy = ix[i], iy[i]
        cnt[cx, cy] += 1

        # Welford update for std
        delta = z[i] - mean[cx, cy]
        mean[cx, cy] += delta / cnt[cx, cy]
        delta2 = z[i] - mean[cx, cy]
        m2[cx, cy] += delta * delta2

        if z[i] < zmin[cx, cy]: zmin[cx, cy] = z[i]
        if z[i] > zmax[cx, cy]: zmax[cx, cy] = z[i]

        if sumi is not None:
            sumi[cx, cy] += intensity[i]

    valid = cnt >= MIN_POINTS_PER_CELL
    z_std = np.zeros((nx, ny), dtype=np.float32)
    z_std[valid] = np.sqrt(m2[valid] / np.maximum(cnt[valid] - 1, 1))

    z_rng = np.zeros((nx, ny), dtype=np.float32)
    z_rng[valid] = (zmax[valid] - zmin[valid])

    if sumi is not None:
        i_mean = np.ones((nx, ny), dtype=np.float32)
        i_mean[valid] = sumi[valid] / cnt[valid]
    else:
        i_mean = None

    # Water cell mask
    water_cell = valid & (z_std <= Z_STD_MAX) & (z_rng <= Z_RANGE_MAX)
    if i_mean is not None:
        water_cell = water_cell & (i_mean <= INTENSITY_MAX)

    print("Water cells:", int(water_cell.sum()), "of", int(valid.sum()), "valid cells")

    # Connected components
    structure = np.ones((3, 3), dtype=np.int8)
    lbl, ncomp = label(water_cell, structure=structure)
    print("Components:", ncomp)

    out_cls = cls.copy()
    if ncomp == 0:
        print("No water detected. Saved unchanged output.")
        out = laspy.LasData(header=las.header)
        out.points = las.points
        out.classification = out_cls.astype(np.uint8)
        out.write(OUT_LAZ)
        print("Saved:", OUT_LAZ)
        return

    # Component info: (comp_id, cells, extent_m)
    comp_info = []
    for comp in range(1, ncomp + 1):
        pts = np.where(lbl == comp)
        if len(pts[0]) == 0:
            continue
        ex = (np.ptp(pts[0]) + 1) * GRID
        ey = (np.ptp(pts[1]) + 1) * GRID
        extent = max(ex, ey)
        comp_info.append((comp, len(pts[0]), extent))

    comp_info.sort(key=lambda t: t[1], reverse=True)

    # Sea = largest component
    sea_comp, sea_cells, sea_extent = comp_info[0]
    print("Sea comp:", sea_comp, "cells:", sea_cells, "extent(m):", sea_extent)

    # per point component id
    cell_comp = lbl[ix, iy]
    sea_mask = (cell_comp == sea_comp)
    out_cls[sea_mask] = SEA_CLASS

    inland_pts = 0
    for comp, cells, extent in comp_info[1:]:
        if extent >= MIN_INLAND_M:
            m = (cell_comp == comp)
            out_cls[m] = LAKE_CLASS
            inland_pts += int(m.sum())

    print("Sea points:", int(sea_mask.sum()))
    print("Inland water points:", inland_pts)

    out = laspy.LasData(header=las.header)
    out.points = las.points
    out.classification = out_cls.astype(np.uint8)
    out.write(OUT_LAZ)

    u, c = np.unique(out_cls, return_counts=True)
    print("\nSaved:", OUT_LAZ)
    for a, b in zip(u, c):
        print(f"Class {int(a):2d}: {int(b):,}")

if __name__ == "__main__":
    main()