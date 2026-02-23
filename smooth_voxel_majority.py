import os
import numpy as np
import laspy
from tqdm import tqdm

IN_LAZ  = r"d:\lidarrrrr\anbu\classified_output.laz"
OUT_LAZ = r"d:\lidarrrrr\anbu\classified_output_smooth.laz"

VOXEL = 0.8   # meters (try 0.5, 0.8, 1.0)
KEEP_CLASSES = np.array([1,2,3,4,5,6], dtype=np.int32)

def main():
    las = laspy.read(IN_LAZ)
    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
    cls = np.array(las.classification, dtype=np.int32)

    # Safety clamp
    bad = ~np.isin(cls, KEEP_CLASSES)
    if bad.any():
        cls[bad] = 1

    # voxel keys
    mn = xyz.min(axis=0)
    ijk = np.floor((xyz - mn) / VOXEL).astype(np.int32)

    # pack ijk -> single key (hash)
    key = (ijk[:,0].astype(np.int64) << 42) ^ (ijk[:,1].astype(np.int64) << 21) ^ ijk[:,2].astype(np.int64)

    order = np.argsort(key)
    key_s = key[order]
    cls_s = cls[order]

    out_cls = cls.copy()

    # run-length groups
    starts = np.r_[0, np.where(key_s[1:] != key_s[:-1])[0] + 1]
    ends   = np.r_[starts[1:], len(key_s)]

    for s, e in tqdm(zip(starts, ends), total=len(starts), desc="Voxel majority"):
        group_idx_sorted = order[s:e]
        labels = cls[group_idx_sorted]
        vals, cnts = np.unique(labels, return_counts=True)
        maj = vals[np.argmax(cnts)]
        out_cls[group_idx_sorted] = maj

    out = laspy.LasData(header=las.header)
    out.points = las.points
    out.classification = out_cls.astype(np.uint8)
    out.write(OUT_LAZ)

    u, c = np.unique(out_cls, return_counts=True)
    print("Saved:", OUT_LAZ)
    for a, b in zip(u, c):
        print(f"  Class {int(a):2d}: {int(b):,}")

if __name__ == "__main__":
    main()