import os
import numpy as np
import laspy
from tqdm import tqdm

# -----------------------------
# CONFIG (edit paths if needed)
# -----------------------------
IN_LAZ  = r"d:\lidarrrrr\anbu\classified_output.laz"
OUT_LAZ = r"d:\lidarrrrr\anbu\classified_output_HAGfix.laz"

# HAG rule
VEG_CLASS = 3
GROUND_CLASS = 2
DEFAULT_CLASS = 1
HAG_VEG_MIN_M = 1.0   # vegetation only if >= 1m above ground

# Ground model settings
GRID_M = 1.0          # ground grid resolution (meters). Try 0.5 / 1.0 / 2.0
FILL_SEARCH_RADIUS = 3  # grid cells radius to fill missing ground cells

# Optional: clean veg near buildings
ENABLE_BUILDING_CLEAN = True
BUILDING_CLASS = 6
BUILDING_RADIUS_M = 0.8    # if veg within this XY distance to a building point
BUILDING_HAG_MAX_M = 5.0   # only apply this correction below this height


def build_ground_grid(x, y, z, cls, grid_m):
    """Create a grid ground surface using class 2 points (min z per cell)."""
    m = (cls == GROUND_CLASS)
    if m.sum() < 1000:
        raise RuntimeError("Not enough ground points (Class 2) to build ground surface.")

    gx = x[m]; gy = y[m]; gz = z[m]

    xmin, ymin = gx.min(), gy.min()
    ix = np.floor((gx - xmin) / grid_m).astype(np.int32)
    iy = np.floor((gy - ymin) / grid_m).astype(np.int32)

    nx = int(ix.max()) + 1
    ny = int(iy.max()) + 1

    # Use +inf grid then take minimum z per cell
    grid = np.full((nx, ny), np.inf, dtype=np.float32)

    # Min aggregation (fast)
    flat_idx = ix.astype(np.int64) * ny + iy.astype(np.int64)
    order = np.argsort(flat_idx)
    flat_s = flat_idx[order]
    gz_s = gz[order]

    # group by flat cell index
    starts = np.r_[0, np.where(flat_s[1:] != flat_s[:-1])[0] + 1]
    ends   = np.r_[starts[1:], len(flat_s)]

    for s, e in zip(starts, ends):
        fi = flat_s[s]
        cx = int(fi // ny)
        cy = int(fi % ny)
        grid[cx, cy] = float(np.min(gz_s[s:e]))

    return grid, xmin, ymin


def fill_missing_ground(grid, search_radius=3):
    """Fill inf cells by nearest valid neighbor within a small radius."""
    nx, ny = grid.shape
    filled = grid.copy()
    valid = np.isfinite(grid)

    # If too many missing, still try filling but warn
    miss = (~valid).sum()
    if miss > 0:
        print(f"Ground grid missing cells: {miss:,} (will fill)")

    # Precompute valid coords
    vx, vy = np.where(valid)
    if len(vx) == 0:
        raise RuntimeError("Ground grid has no valid cells after creation.")

    # For each missing cell, search in growing radius up to search_radius
    # (small radius only; fast enough)
    for r in range(1, search_radius + 1):
        missing = np.where(~np.isfinite(filled))
        if len(missing[0]) == 0:
            break
        mx, my = missing
        for i in range(len(mx)):
            cx, cy = mx[i], my[i]
            x0, x1 = max(0, cx - r), min(nx, cx + r + 1)
            y0, y1 = max(0, cy - r), min(ny, cy + r + 1)
            win = filled[x0:x1, y0:y1]
            if np.isfinite(win).any():
                filled[cx, cy] = float(np.nanmin(np.where(np.isfinite(win), win, np.nan)))

    # Any still missing -> fill with global min (safe fallback)
    still = ~np.isfinite(filled)
    if still.any():
        gmin = np.nanmin(np.where(np.isfinite(filled), filled, np.nan))
        filled[still] = gmin

    return filled


def ground_z_for_points(x, y, grid, xmin, ymin, grid_m):
    """Lookup ground z by nearest grid cell."""
    ix = np.floor((x - xmin) / grid_m).astype(np.int32)
    iy = np.floor((y - ymin) / grid_m).astype(np.int32)

    ix = np.clip(ix, 0, grid.shape[0] - 1)
    iy = np.clip(iy, 0, grid.shape[1] - 1)
    return grid[ix, iy]


def building_clean(x, y, cls, hag):
    """Optional rule: if veg is very close to building points, flip veg->building."""
    veg_mask = (cls == VEG_CLASS)
    b_mask = (cls == BUILDING_CLASS)
    if veg_mask.sum() == 0 or b_mask.sum() == 0:
        return cls

    # Only fix low-ish veg near buildings (avoid converting real trees beside buildings)
    veg_mask = veg_mask & (hag <= BUILDING_HAG_MAX_M)
    if veg_mask.sum() == 0:
        return cls

    # Build coarse spatial hash for building points (fast)
    cell = BUILDING_RADIUS_M
    bx = x[b_mask]; by = y[b_mask]

    bx0 = bx.min(); by0 = by.min()
    bix = np.floor((bx - bx0) / cell).astype(np.int32)
    biy = np.floor((by - by0) / cell).astype(np.int32)

    # map (ix,iy) -> list (store packed keys)
    key = (bix.astype(np.int64) << 32) ^ biy.astype(np.int64)
    key_sort = np.argsort(key)
    key_s = key[key_sort]
    bx_s = bx[key_sort]
    by_s = by[key_sort]

    # index ranges for each key
    starts = np.r_[0, np.where(key_s[1:] != key_s[:-1])[0] + 1]
    ends   = np.r_[starts[1:], len(key_s)]
    uniq_keys = key_s[starts]
    # build dict: key -> (start,end)
    ranges = {int(k): (int(s), int(e)) for k, s, e in zip(uniq_keys, starts, ends)}

    vx = x[veg_mask]; vy = y[veg_mask]
    vix = np.floor((vx - bx0) / cell).astype(np.int32)
    viy = np.floor((vy - by0) / cell).astype(np.int32)

    cls_out = cls.copy()
    veg_idx = np.where(veg_mask)[0]

    r2 = BUILDING_RADIUS_M * BUILDING_RADIUS_M

    for i in tqdm(range(len(veg_idx)), desc="Building clean", leave=False):
        ix, iy = int(vix[i]), int(viy[i])

        # search neighbor cells (3x3)
        near = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                k = ((ix + dx) << 32) ^ (iy + dy)
                if int(k) not in ranges:
                    continue
                s, e = ranges[int(k)]
                dxs = bx_s[s:e] - vx[i]
                dys = by_s[s:e] - vy[i]
                if np.any(dxs * dxs + dys * dys <= r2):
                    near = True
                    break
            if near:
                break

        if near:
            cls_out[veg_idx[i]] = BUILDING_CLASS

    return cls_out


def main():
    if not os.path.exists(IN_LAZ):
        raise FileNotFoundError(IN_LAZ)

    print("Reading:", IN_LAZ)
    las = laspy.read(IN_LAZ)

    x = np.array(las.x, dtype=np.float32)
    y = np.array(las.y, dtype=np.float32)
    z = np.array(las.z, dtype=np.float32)
    cls = np.array(las.classification, dtype=np.int32)

    print("Points:", f"{len(cls):,}")
    print("Unique classes:", np.unique(cls))

    # 1) Ground grid
    print(f"\nBuilding ground grid from Class {GROUND_CLASS} at GRID={GRID_M}m ...")
    grid, xmin, ymin = build_ground_grid(x, y, z, cls, GRID_M)
    grid = fill_missing_ground(grid, search_radius=FILL_SEARCH_RADIUS)

    # 2) HAG
    gz = ground_z_for_points(x, y, grid, xmin, ymin, GRID_M)
    hag = (z - gz).astype(np.float32)

    # 3) Vegetation rule: no veg below 1m
    veg_mask = (cls == VEG_CLASS)
    fix_mask = veg_mask & (hag < HAG_VEG_MIN_M)
    print(f"\nVeg points: {veg_mask.sum():,}")
    print(f"Veg < {HAG_VEG_MIN_M}m (to Default {DEFAULT_CLASS}): {fix_mask.sum():,}")

    cls2 = cls.copy()
    cls2[fix_mask] = DEFAULT_CLASS

    # Optional: veg near buildings -> building
    if ENABLE_BUILDING_CLEAN:
        print("\nOptional: cleaning vegetation near buildings...")
        cls2 = building_clean(x, y, cls2, hag)

    # Save output
    out = laspy.LasData(header=las.header)
    out.points = las.points
    out.classification = cls2.astype(np.uint8)
    out.write(OUT_LAZ)

    u, c = np.unique(cls2, return_counts=True)
    print("\nSaved:", OUT_LAZ)
    print("Class counts:")
    for a, b in zip(u, c):
        print(f"  Class {int(a):2d}: {int(b):,}")


if __name__ == "__main__":
    main()