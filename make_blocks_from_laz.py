import os, json, math, random
import numpy as np
import laspy
from tqdm import tqdm

# ---------------------------
# CONFIG (EDIT PATHS)
# ---------------------------
CONFIG = {
    "classified_files": [
        r"d:\lidarrrrr\anbu\training_labeled\DX3011148 ULMIANO000001.laz",
        r"d:\lidarrrrr\anbu\training_labeled\DX3011148 ULMIANO000003.laz",
        r"d:\lidarrrrr\anbu\training_labeled\DX3011148 ULMIANO000004.laz",
        r"d:\lidarrrrr\anbu\training_labeled\DX3011148 ULMIANO000005.laz",
        r"d:\lidarrrrr\anbu\training_labeled\pt013390.laz",
    ],

    "out_root": r"d:\lidarrrrr\anbu\randla_dataset",
    "block_points": 4096,           # ✅ safe for RTX 3050
    "blocks_per_file": 250,         # increase if you want more data (e.g., 500)
    "val_ratio": 0.15,              # 15% validation blocks
    "use_features": True,           # intensity/returns
    "seed": 42,

    # keep only 2–6 for training, map others -> 1
    "keep_classes": [2, 3, 4, 5, 6],
    "map_others_to": 1,
}

# Non-standard -> standard (edit if your data has more)
CLASS_REMAP = {
    0: 1,
    18: 6,
    19: 5,
    20: 5,
    65: 2,
    66: 2,
    67: 3,
    68: 4,
    69: 5,
}

def ensure_dirs(root):
    train_dir = os.path.join(root, "train")
    val_dir   = os.path.join(root, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    return train_dir, val_dir

def read_laz(filepath):
    las = laspy.read(filepath)
    dims = set(las.point_format.dimension_names)

    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
    cls = np.array(las.classification, dtype=np.int32)

    # remap non-standard
    for a, b in CLASS_REMAP.items():
        m = (cls == a)
        if m.any():
            cls[m] = b

    # map not-in-keep to 1
    keep = set(CONFIG["keep_classes"])
    cls = np.where(np.isin(cls, list(keep)), cls, CONFIG["map_others_to"]).astype(np.int32)

    feats = None
    if CONFIG["use_features"]:
        intensity = np.array(las.intensity, dtype=np.float32) if "intensity" in dims else np.zeros(len(cls), np.float32)
        rn       = np.array(las.return_number, dtype=np.float32) if "return_number" in dims else np.ones(len(cls), np.float32)
        nr       = np.array(las.number_of_returns, dtype=np.float32) if "number_of_returns" in dims else np.ones(len(cls), np.float32)
        ret_ratio = rn / (nr + 1e-6)
        # normalize intensity
        if intensity.max() > intensity.min():
            intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())
        feats = np.column_stack([intensity, ret_ratio, nr]).astype(np.float32)

    return xyz, feats, cls

def sample_block_indices(xyz, cls, n_points, focus_classes):
    """
    Sample a block centered near points from focus_classes to balance classes.
    Uses a simple center + nearest strategy (fast, good enough for dataset creation).
    """
    # choose a center from a target class if possible
    mask = np.isin(cls, focus_classes)
    if mask.any():
        center_idx = np.random.choice(np.where(mask)[0])
    else:
        center_idx = np.random.randint(0, len(cls))

    center_xy = xyz[center_idx, :2]  # x,y

    # compute squared dist in XY (fast)
    d2 = np.sum((xyz[:, :2] - center_xy[None, :])**2, axis=1)
    idx = np.argpartition(d2, n_points)[:n_points]
    return idx

def make_blocks_for_file(filepath, blocks_per_file, block_points, train_dir, val_dir, val_ratio):
    xyz, feats, cls = read_laz(filepath)

    # we only want blocks that contain at least some keep classes (2-6)
    keep = np.array(CONFIG["keep_classes"], dtype=np.int32)
    n = len(cls)
    if n < block_points:
        return 0, 0

    base = os.path.splitext(os.path.basename(filepath))[0].replace(" ", "_")
    made_train, made_val = 0, 0

    for b in tqdm(range(blocks_per_file), desc=f"Blocks {base}", leave=False):
        idx = sample_block_indices(xyz, cls, block_points, focus_classes=keep)

        blk_xyz = xyz[idx]
        blk_cls = cls[idx]

        # Require some keep labels in block (avoid all class 1 blocks)
        if not np.isin(blk_cls, keep).any():
            continue

        # Normalize block: center XY, Z to min (helps training)
        blk_xyz = blk_xyz.copy()
        blk_xyz[:, 0] -= blk_xyz[:, 0].mean()
        blk_xyz[:, 1] -= blk_xyz[:, 1].mean()
        blk_xyz[:, 2] -= blk_xyz[:, 2].min()

        blk_feats = feats[idx] if feats is not None else None

        out = {
            "points": blk_xyz.astype(np.float32),
            "labels": blk_cls.astype(np.int32)
        }
        if blk_feats is not None:
            out["features"] = blk_feats.astype(np.float32)

        is_val = (random.random() < val_ratio)
        out_dir = val_dir if is_val else train_dir

        fname = f"{base}_b{b:04d}.npz"
        np.savez_compressed(os.path.join(out_dir, fname), **out)

        if is_val:
            made_val += 1
        else:
            made_train += 1

    return made_train, made_val

def main():
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    train_dir, val_dir = ensure_dirs(CONFIG["out_root"])

    total_train, total_val = 0, 0
    print("Creating blocks dataset...")
    print("Output:", CONFIG["out_root"])

    for f in CONFIG["classified_files"]:
        if not os.path.exists(f):
            print("Missing:", f)
            continue
        t, v = make_blocks_for_file(
            f, CONFIG["blocks_per_file"], CONFIG["block_points"],
            train_dir, val_dir, CONFIG["val_ratio"]
        )
        total_train += t
        total_val   += v
        print(f"✔ {os.path.basename(f)} -> train {t}, val {v}")

    meta = {
        "block_points": CONFIG["block_points"],
        "use_features": CONFIG["use_features"],
        "keep_classes": CONFIG["keep_classes"],
        "mapped_other_class": CONFIG["map_others_to"],
        "class_remap": CLASS_REMAP,
        "train_blocks": total_train,
        "val_blocks": total_val
    }
    with open(os.path.join(CONFIG["out_root"], "meta.json"), "w") as fp:
        json.dump(meta, fp, indent=2)

    print("\n✅ Done!")
    print("Train blocks:", total_train)
    print("Val blocks  :", total_val)
    print("Meta saved  :", os.path.join(CONFIG['out_root'], "meta.json"))

if __name__ == "__main__":
    main()