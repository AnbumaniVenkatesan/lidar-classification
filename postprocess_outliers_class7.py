import os
import numpy as np
import laspy
from tqdm import tqdm

IN_LAZ  = r"d:\lidarrrrr\anbu\classified_output_HAGfix.laz"
OUT_LAZ = r"d:\lidarrrrr\anbu\classified_output_HAGfix_out7.laz"

GRID_M = 1.0

# Outlier rules
MIN_CELL_COUNT = 3         # cells with <= this count are "isolated"
Z_ROBUST_THRESH = 2.5      # robust z-score threshold inside each cell

DEFAULT_CLASS = 1
OUTLIER_CLASS = 7

# Keep these protected (we won't change them to outliers unless extremely isolated)
PROTECT_CLASSES = set([2, 6])   # ground & buildings


def main():
    if not os.path.exists(IN_LAZ):
        raise FileNotFoundError(IN_LAZ)

    print("Reading:", IN_LAZ)
    las = laspy.read(IN_LAZ)

    x = np.array(las.x, dtype=np.float32)
    y = np.array(las.y, dtype=np.float32)
    z = np.array(las.z, dtype=np.float32)
    cls = np.array(las.classification, dtype=np.int32)

    n = len(cls)
    print("Points:", f"{n:,}")
    print("Classes:", np.unique(cls))

    xmin, ymin = x.min(), y.min()
    ix = np.floor((x - xmin) / GRID_M).astype(np.int32)
    iy = np.floor((y - ymin) / GRID_M).astype(np.int32)

    ny = int(iy.max()) + 1
    cell = ix.astype(np.int64) * ny + iy.astype(np.int64)

    order = np.argsort(cell)
    cell_s = cell[order]
    z_s = z[order]
    cls_s = cls[order]

    starts = np.r_[0, np.where(cell_s[1:] != cell_s[:-1])[0] + 1]
    ends   = np.r_[starts[1:], len(cell_s)]

    out_cls = cls.copy()
    outlier_mask = np.zeros(n, dtype=bool)

    print("Detecting outliers per grid cell...")
    for s, e in tqdm(zip(starts, ends), total=len(starts), desc="Outliers"):
        idx_sorted = order[s:e]
        m = e - s

        # isolated cell
        if m <= MIN_CELL_COUNT:
            # do not override protected classes unless they are not protected
            protect = np.isin(out_cls[idx_sorted], list(PROTECT_CLASSES))
            mark = ~protect
            if mark.any():
                outlier_mask[idx_sorted[mark]] = True
            continue

        # robust z-score inside the cell
        zz = z_s[s:e]
        med = np.median(zz)
        mad = np.median(np.abs(zz - med)) + 1e-6
        rz = np.abs(zz - med) / (1.4826 * mad)

        bad = rz > Z_ROBUST_THRESH

        # don't convert protected classes unless extremely bad
        if bad.any():
            idx_bad = idx_sorted[bad]
            protect = np.isin(out_cls[idx_bad], list(PROTECT_CLASSES))
            # if protected, require stronger condition
            rz_bad = rz[bad]
            strong = rz_bad > (Z_ROBUST_THRESH * 1.5)
            mark = (~protect) | strong
            outlier_mask[idx_bad[mark]] = True

    # Apply: outliers -> class 7 (but only from default/veg/water types)
    # We'll avoid breaking ground/building unless they were marked strong.
    changed = outlier_mask.sum()
    print("Outliers flagged:", f"{changed:,}")
    out_cls[outlier_mask] = OUTLIER_CLASS

    out = laspy.LasData(header=las.header)
    out.points = las.points
    out.classification = out_cls.astype(np.uint8)
    out.write(OUT_LAZ)

    u, c = np.unique(out_cls, return_counts=True)
    print("\nSaved:", OUT_LAZ)
    for a, b in zip(u, c):
        print(f"  Class {int(a):2d}: {int(b):,}")

if __name__ == "__main__":
    main()