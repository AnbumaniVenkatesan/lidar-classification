import os
import numpy as np
import laspy
from tqdm import tqdm

IN_LAZ  = r"d:\lidarrrrr\anbu\classified_output_SCOPE_FINAL.laz"
OUT_LAZ = r"d:\lidarrrrr\anbu\classified_output_SCOPE_FINAL_overlap12.laz"

OVERLAP_CLASS = 12

# Grid used to detect overlap zones between strips
GRID_M = 2.0

# "Central corridor" definition per strip:
# Points within this fraction of the strip cross-track width are kept as non-overlap.
# Example: 0.35 keeps ~central 70% band (±35%).
CENTRAL_FRAC = 0.35

# Do not overwrite these classes with overlap (safety)
PROTECT_CLASSES = {7}  # keep outliers as outliers; adjust if you want overlap to override everything

def main():
    if not os.path.exists(IN_LAZ):
        raise FileNotFoundError(IN_LAZ)

    print("Reading:", IN_LAZ)
    las = laspy.read(IN_LAZ)

    dims = set(las.point_format.dimension_names)
    if "point_source_id" not in dims:
        raise RuntimeError("point_source_id not found. Cannot compute overlap by strip.")

    x = np.array(las.x, dtype=np.float32)
    y = np.array(las.y, dtype=np.float32)
    cls = np.array(las.classification, dtype=np.int32)
    sid = np.array(las.point_source_id, dtype=np.int32)

    n = len(cls)
    print("Points:", f"{n:,}")
    print("Unique strip ids:", len(np.unique(sid)))

    # -------------------------
    # 1) Build grid cell ids
    # -------------------------
    xmin, ymin = x.min(), y.min()
    ix = np.floor((x - xmin) / GRID_M).astype(np.int32)
    iy = np.floor((y - ymin) / GRID_M).astype(np.int32)

    ny = int(iy.max()) + 1
    cell = ix.astype(np.int64) * ny + iy.astype(np.int64)

    # Sort by (cell, strip) to detect overlap cells
    order = np.lexsort((sid, cell))
    cell_s = cell[order]
    sid_s  = sid[order]

    # -------------------------
    # 2) Find overlap cells (cells that contain >=2 different strips)
    # -------------------------
    # First group by cell
    starts = np.r_[0, np.where(cell_s[1:] != cell_s[:-1])[0] + 1]
    ends   = np.r_[starts[1:], len(cell_s)]

    overlap_cell = np.zeros(int(cell.max()) + 1, dtype=bool)

    for s, e in tqdm(zip(starts, ends), total=len(starts), desc="Detect overlap cells"):
        if sid_s[s:e].min() != sid_s[s:e].max():  # fast check: more than one strip
            overlap_cell[int(cell_s[s])] = True

    # per-point overlap zone mask (by cell)
    in_overlap_zone = overlap_cell[cell.astype(np.int64)]
    print("Points in overlap zone:", f"{int(in_overlap_zone.sum()):,}")

    if in_overlap_zone.sum() == 0:
        print("No overlap zones found. Saving unchanged.")
        out = laspy.LasData(header=las.header)
        out.points = las.points
        out.classification = cls.astype(np.uint8)
        out.write(OUT_LAZ)
        print("Saved:", OUT_LAZ)
        return

    # -------------------------
    # 3) Central corridor per strip (PCA direction)
    #    Compute cross-track coordinate t and mark central band
    # -------------------------
    central_mask = np.zeros(n, dtype=bool)

    strip_ids = np.unique(sid)
    for s_id in tqdm(strip_ids, desc="Central corridor per strip"):
        m = (sid == s_id)
        if m.sum() < 10000:
            # small strip → treat all as central
            central_mask[m] = True
            continue

        # PCA on XY to get along-track axis
        pts = np.column_stack([x[m], y[m]]).astype(np.float64)
        mu = pts.mean(axis=0)
        pts0 = pts - mu
        cov = (pts0.T @ pts0) / max(len(pts0) - 1, 1)
        w, v = np.linalg.eigh(cov)
        along = v[:, np.argmax(w)]  # principal direction
        # cross-track axis (perpendicular)
        cross = np.array([-along[1], along[0]])

        # cross-track coordinate
        t = (pts0 @ cross)

        # robust strip width estimate using percentiles
        p05, p95 = np.percentile(t, [5, 95])
        half_width = (p95 - p05) / 2.0
        if half_width <= 1e-6:
            central_mask[m] = True
            continue

        # central band threshold
        thr = CENTRAL_FRAC * (2.0 * half_width)  # fraction of full width
        # center is median
        med = np.median(t)
        central = np.abs(t - med) <= thr
        # write back
        idx = np.where(m)[0]
        central_mask[idx] = central

    # -------------------------
    # 4) Apply overlap class:
    #    in overlap zone AND NOT central => class 12
    # -------------------------
    out_cls = cls.copy()

    # do not change protected classes
    can_change = ~np.isin(out_cls, list(PROTECT_CLASSES))

    set12 = in_overlap_zone & (~central_mask) & can_change
    out_cls[set12] = OVERLAP_CLASS

    print("Overlap points set to Class 12:", f"{int(set12.sum()):,}")

    # Save
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