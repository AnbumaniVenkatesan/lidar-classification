import os
import numpy as np
import laspy
from tqdm import tqdm
from scipy.ndimage import label

IN_LAZ  = r"d:\lidarrrrr\anbu\classified_output_HAGfix_out7.laz"
OUT_LAZ = r"d:\lidarrrrr\anbu\classified_output_WATERFIX.laz"

SEA_CLASS   = 9
LAKE_CLASS  = 13
DEFAULT     = 1

GRID = 1.0           # meters
MIN_INLAND_M = 10.0  # only inland components >= 10m

# water detection thresholds (tune if needed)
MIN_POINTS_PER_CELL = 5
MAX_Z_DEV = 0.20          # meters: point must be within this of cell mean plane
INTENSITY_MAX = 0.25      # lower intensity often = water (dataset-dependent)

print("Reading:", IN_LAZ)
las = laspy.read(IN_LAZ)

x = np.array(las.x, dtype=np.float32)
y = np.array(las.y, dtype=np.float32)
z = np.array(las.z, dtype=np.float32)
cls = np.array(las.classification, dtype=np.int32)

dims = set(las.point_format.dimension_names)
has_intensity = "intensity" in dims
print("Intensity available:", has_intensity)

if has_intensity:
    intensity = np.array(las.intensity, dtype=np.float32)
    intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-6)

xmin, ymin = x.min(), y.min()
ix = np.floor((x - xmin) / GRID).astype(np.int32)
iy = np.floor((y - ymin) / GRID).astype(np.int32)
nx = int(ix.max()) + 1
ny = int(iy.max()) + 1

grid_sumz = np.zeros((nx, ny), dtype=np.float64)
grid_cnt  = np.zeros((nx, ny), dtype=np.int32)
grid_sumi = np.zeros((nx, ny), dtype=np.float64) if has_intensity else None

print("Building cell stats...")
for i in tqdm(range(len(x)), desc="Cell stats"):
    cx, cy = ix[i], iy[i]
    grid_sumz[cx, cy] += float(z[i])
    grid_cnt[cx, cy]  += 1
    if has_intensity:
        grid_sumi[cx, cy] += float(intensity[i])

valid = grid_cnt >= MIN_POINTS_PER_CELL
grid_meanz = np.zeros((nx, ny), dtype=np.float32)
grid_meanz[valid] = (grid_sumz[valid] / grid_cnt[valid]).astype(np.float32)

if has_intensity:
    grid_meani = np.ones((nx, ny), dtype=np.float32)
    grid_meani[valid] = (grid_sumi[valid] / grid_cnt[valid]).astype(np.float32)

# Start with cells that are valid
water_cell = valid.copy()

print("Flatness test...")
# If any point in a cell deviates too much from mean plane → not water
# (fast approximate test)
for i in tqdm(range(len(x)), desc="Flat test"):
    cx, cy = ix[i], iy[i]
    if not water_cell[cx, cy]:
        continue
    if abs(z[i] - grid_meanz[cx, cy]) > MAX_Z_DEV:
        water_cell[cx, cy] = False

if has_intensity:
    water_cell = water_cell & (grid_meani < INTENSITY_MAX)

print("Connected components on water cells...")
structure = np.ones((3, 3), dtype=np.int8)
lbl, ncomp = label(water_cell, structure=structure)
print("Components:", ncomp)

# If no components, just save input as output
out_cls = cls.copy()
if ncomp == 0:
    out = laspy.LasData(header=las.header)
    out.points = las.points
    out.classification = out_cls.astype(np.uint8)
    out.write(OUT_LAZ)
    print("No water detected. Saved:", OUT_LAZ)
    raise SystemExit

# Compute component sizes and extents in meters
comp_info = []
for comp in range(1, ncomp + 1):
    pts = np.where(lbl == comp)
    if len(pts[0]) == 0:
        continue
    # extent in meters (grid units * GRID)
    ex = (np.ptp(pts[0]) + 1) * GRID
    ey = (np.ptp(pts[1]) + 1) * GRID
    extent = max(ex, ey)
    area_cells = len(pts[0])  # number of grid cells in component
    comp_info.append((comp, area_cells, extent))

# Sea = largest component by area (cells)
comp_info.sort(key=lambda t: t[1], reverse=True)
sea_comp = comp_info[0][0]
print("Sea component id:", sea_comp, "cells:", comp_info[0][1], "extent(m):", comp_info[0][2])

# Assign per-point water class using its cell component id
cell_comp = lbl[ix, iy]  # per point component id (0 if not water cell)

# Sea
sea_mask = (cell_comp == sea_comp)
out_cls[sea_mask] = SEA_CLASS

# Inland water: other components with extent >= 10m
inland_assigned = 0
for comp, area_cells, extent in comp_info[1:]:
    if extent >= MIN_INLAND_M:
        m = (cell_comp == comp)
        out_cls[m] = LAKE_CLASS
        inland_assigned += int(m.sum())

print("Sea points:", int(sea_mask.sum()))
print("Inland water points:", inland_assigned)

out = laspy.LasData(header=las.header)
out.points = las.points
out.classification = out_cls.astype(np.uint8)
out.write(OUT_LAZ)

u, c = np.unique(out_cls, return_counts=True)
print("\nSaved:", OUT_LAZ)
for a, b in zip(u, c):
    print(f"Class {int(a):2d}: {int(b):,}")