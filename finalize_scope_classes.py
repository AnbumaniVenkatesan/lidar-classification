import os
import numpy as np
import laspy
from tqdm import tqdm

IN_LAZ  = r"d:\lidarrrrr\anbu\classified_output_WATERFIX_v3.laz"
OUT_LAZ = r"d:\lidarrrrr\anbu\classified_output_SCOPE_FINAL.laz"

# Target classes required by your scope
TARGET = {1,2,3,6,7,9,10,12,13}

# Vegetation merging
VEG_CLASSES_TO_MERGE = {4,5}   # -> 3
VEG_CLASS = 3

# Bridge detection params (rule-based)
ENABLE_BRIDGE = True
BRIDGE_CLASS = 10
WATER_CLASSES = {9,13}
GROUND_CLASS = 2

GRID = 2.0                 # meters grid for bridge detection
BRIDGE_MIN_HAG = 2.0       # bridge must be above local ground by at least 2m
BRIDGE_MAX_ZSTD = 0.25     # bridge deck is planar
BRIDGE_MIN_CELLS = 30      # minimum connected cells to be a bridge patch (tune)

def main():
    if not os.path.exists(IN_LAZ):
        raise FileNotFoundError(IN_LAZ)

    print("Reading:", IN_LAZ)
    las = laspy.read(IN_LAZ)
    x = np.array(las.x, dtype=np.float32)
    y = np.array(las.y, dtype=np.float32)
    z = np.array(las.z, dtype=np.float32)
    cls = np.array(las.classification, dtype=np.int32)

    print("Unique classes (before):", np.unique(cls))

    # 1) Merge veg 4/5 -> 3
    m = np.isin(cls, list(VEG_CLASSES_TO_MERGE))
    if m.any():
        cls[m] = VEG_CLASS
        print(f"Merged {m.sum():,} points from {VEG_CLASSES_TO_MERGE} -> {VEG_CLASS}")

    # 2) Bridge detection (optional)
    if ENABLE_BRIDGE:
        print("\nDetecting bridges (rule-based)...")
        # Build grid indices
        xmin, ymin = x.min(), y.min()
        ix = np.floor((x - xmin) / GRID).astype(np.int32)
        iy = np.floor((y - ymin) / GRID).astype(np.int32)
        nx = int(ix.max()) + 1
        ny = int(iy.max()) + 1

        # For each cell: count, mean z, std z, min ground z, water flag
        cnt  = np.zeros((nx, ny), dtype=np.int32)
        mean = np.zeros((nx, ny), dtype=np.float32)
        m2   = np.zeros((nx, ny), dtype=np.float32)

        gmin = np.full((nx, ny), np.inf, dtype=np.float32)  # min ground z per cell
        water_cell = np.zeros((nx, ny), dtype=bool)

        # 1 pass accumulation
        for i in tqdm(range(len(cls)), desc="Bridge grid"):
            cx, cy = ix[i], iy[i]
            cnt[cx, cy] += 1
            dz = z[i] - mean[cx, cy]
            mean[cx, cy] += dz / cnt[cx, cy]
            dz2 = z[i] - mean[cx, cy]
            m2[cx, cy] += dz * dz2

            if cls[i] == GROUND_CLASS:
                if z[i] < gmin[cx, cy]:
                    gmin[cx, cy] = z[i]

            if cls[i] in WATER_CLASSES:
                water_cell[cx, cy] = True

        valid = cnt > 5
        zstd = np.zeros((nx, ny), dtype=np.float32)
        zstd[valid] = np.sqrt(m2[valid] / np.maximum(cnt[valid] - 1, 1))

        # Candidate bridge cells:
        # - near water
        # - planar
        # - above local ground if ground exists
        cand = valid & water_cell & (zstd <= BRIDGE_MAX_ZSTD)

        # Require ground to compute HAG; if no ground in cell, skip (safe)
        has_ground = np.isfinite(gmin)
        hag_cell = np.zeros((nx, ny), dtype=np.float32)
        hag_cell[has_ground] = mean[has_ground] - gmin[has_ground]
        cand = cand & has_ground & (hag_cell >= BRIDGE_MIN_HAG)

        # Connected components over candidate cells (simple BFS)
        # We'll label components and keep those big enough
        visited = np.zeros((nx, ny), dtype=bool)
        comp_id = np.zeros((nx, ny), dtype=np.int32)
        comp_sizes = []
        cid = 0

        neighbors = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

        for i in range(nx):
            for j in range(ny):
                if not cand[i,j] or visited[i,j]:
                    continue
                cid += 1
                stack = [(i,j)]
                visited[i,j] = True
                comp_id[i,j] = cid
                size = 0
                while stack:
                    a,b = stack.pop()
                    size += 1
                    for da,db in neighbors:
                        na, nb = a+da, b+db
                        if 0 <= na < nx and 0 <= nb < ny and cand[na,nb] and not visited[na,nb]:
                            visited[na,nb] = True
                            comp_id[na,nb] = cid
                            stack.append((na,nb))
                comp_sizes.append(size)

        if cid == 0:
            print("No bridge candidates found.")
        else:
            comp_sizes = np.array(comp_sizes, dtype=np.int32)
            keep_comps = set(np.where(comp_sizes >= BRIDGE_MIN_CELLS)[0] + 1)  # comps numbered from 1
            print(f"Bridge components found: {cid}, kept: {len(keep_comps)}")

            if len(keep_comps) > 0:
                # mark points whose cell is in kept comps
                pt_comp = comp_id[ix, iy]
                bridge_mask = np.isin(pt_comp, list(keep_comps))

                # Only override default/veg/building? We'll override defaults and veg mostly
                # Avoid changing ground/water/outliers
                safe = ~np.isin(cls, [2,7,9,13])
                final_bridge = bridge_mask & safe
                cls[final_bridge] = BRIDGE_CLASS
                print("Bridge points assigned:", int(final_bridge.sum()))

    # 3) Strict class set: anything not in scope -> Class 1
    bad = ~np.isin(cls, list(TARGET))
    if bad.any():
        cls[bad] = 1
        print("Forced to Class 1 (not in scope):", int(bad.sum()))

    # Save
    out = laspy.LasData(header=las.header)
    out.points = las.points
    out.classification = cls.astype(np.uint8)
    out.write(OUT_LAZ)

    u, c = np.unique(cls, return_counts=True)
    print("\nSaved:", OUT_LAZ)
    for a, b in zip(u, c):
        print(f"Class {int(a):2d}: {int(b):,}")

if __name__ == "__main__":
    main()