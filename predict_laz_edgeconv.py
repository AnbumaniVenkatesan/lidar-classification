import os
import numpy as np
import laspy
import torch
import torch.nn as nn
from tqdm import tqdm

from torch_geometric.nn import DynamicEdgeConv, MLP

# -----------------------
# CONFIG (edit if needed)
# -----------------------
CFG = {
    "model_path": r"d:\lidarrrrr\anbu\dl_models\edgeconv_best.pth",
    "raw_input":  r"d:\lidarrrrr\anbu\DX3035724 S.GIUSTO000001.laz",
    "out_laz":    r"d:\lidarrrrr\anbu\classified_output.laz",

    "block_points": 4096,
    "batch_blocks": 3,          # blocks per GPU batch during inference
    "k": 16,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # If your training blocks used intensity/return features, keep this True.
    # Our block-maker saved features, so this should be True.
    "use_features": True,

    # Safety: only output these classes, else map to 1
    "valid_classes": [1, 2, 3, 4, 5, 6],
}

# -----------------------
# Model (must match training)
# -----------------------
class EdgeSegNet(nn.Module):
    def __init__(self, in_ch, num_classes, k=16):
        super().__init__()
        self.ec1 = DynamicEdgeConv(MLP([2 * in_ch, 64, 64, 64], norm="batch_norm"), k=k, aggr="max")
        self.ec2 = DynamicEdgeConv(MLP([2 * 64, 128, 128], norm="batch_norm"), k=k, aggr="max")
        self.ec3 = DynamicEdgeConv(MLP([2 * 128, 256, 256], norm="batch_norm"), k=k, aggr="max")

        self.head = nn.Sequential(
            nn.Linear(64 + 128 + 256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0),  # inference
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, batch):
        f1 = self.ec1(x, batch)
        f2 = self.ec2(f1, batch)
        f3 = self.ec3(f2, batch)
        feat = torch.cat([f1, f2, f3], dim=1)
        return self.head(feat)

# -----------------------
# Helpers
# -----------------------
def build_features_from_las(las, use_features=True):
    dims = set(las.point_format.dimension_names)
    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)

    if not use_features:
        return xyz, None

    intensity = np.array(las.intensity, dtype=np.float32) if "intensity" in dims else np.zeros(len(xyz), np.float32)
    rn = np.array(las.return_number, dtype=np.float32) if "return_number" in dims else np.ones(len(xyz), np.float32)
    nr = np.array(las.number_of_returns, dtype=np.float32) if "number_of_returns" in dims else np.ones(len(xyz), np.float32)

    ret_ratio = rn / (nr + 1e-6)

    # normalize intensity per file
    if intensity.max() > intensity.min():
        intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())

    feats = np.column_stack([intensity, ret_ratio, nr]).astype(np.float32)
    return xyz, feats

def sample_block_indices_xy(xyz, center_idx, block_points):
    cxy = xyz[center_idx, :2]
    d2 = np.sum((xyz[:, :2] - cxy[None, :]) ** 2, axis=1)
    idx = np.argpartition(d2, block_points)[:block_points]
    return idx

def make_blocks_indices(xyz, block_points):
    """
    Returns a list of index arrays.
    Ensures every point is covered at least once by:
    - shuffling points
    - taking centers from the shuffled list
    """
    n = len(xyz)
    if n <= block_points:
        return [np.arange(n, dtype=np.int64)]

    centers = np.random.permutation(n)
    blocks = []
    step = block_points  # one center per point chunk
    for i in range(0, n, step):
        center_idx = centers[i]
        idx = sample_block_indices_xy(xyz, int(center_idx), block_points)
        blocks.append(idx.astype(np.int64))
    return blocks

def run_inference(model, x_all, blocks, batch_blocks, device):
    """
    x_all: (N, F) float32
    blocks: list of (block_points,) indices
    Returns: predicted label indices (0..C-1) for every point using vote + max prob
    """
    model.eval()
    N = x_all.shape[0]
    num_classes = model.head[-1].out_features

    # voting accumulators
    vote_counts = np.zeros((N, num_classes), dtype=np.int32)

    # process blocks in mini-batches
    with torch.no_grad():
        for s in tqdm(range(0, len(blocks), batch_blocks), desc="Predict blocks"):
            batch = blocks[s:s+batch_blocks]

            # Build concatenated tensor + batch vector
            xs = []
            batch_vec = []
            for bi, idx in enumerate(batch):
                xb = x_all[idx].copy()

                # normalize block similar to training: center XY, set Z min to 0
                xb[:, 0] -= xb[:, 0].mean()
                xb[:, 1] -= xb[:, 1].mean()
                xb[:, 2] -= xb[:, 2].min()

                xs.append(xb)
                batch_vec.append(np.full((len(idx),), bi, dtype=np.int64))

            x_cat = np.concatenate(xs, axis=0).astype(np.float32)
            b_cat = np.concatenate(batch_vec, axis=0).astype(np.int64)

            xt = torch.from_numpy(x_cat).to(device, non_blocking=True)
            bt = torch.from_numpy(b_cat).to(device, non_blocking=True)

            logits = model(xt, bt)
            pred = logits.argmax(1).detach().cpu().numpy()

            # Scatter votes back
            offset = 0
            for bi, idx in enumerate(batch):
                p = pred[offset: offset + len(idx)]
                # add 1 vote for predicted class
                vote_counts[idx, p] += 1
                offset += len(idx)

            del xt, bt, logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    final_idx = vote_counts.argmax(axis=1).astype(np.int64)
    return final_idx

def main():
    np.random.seed(42)
    torch.backends.cudnn.benchmark = True

    if not os.path.exists(CFG["model_path"]):
        raise FileNotFoundError(CFG["model_path"])
    if not os.path.exists(CFG["raw_input"]):
        raise FileNotFoundError(CFG["raw_input"])

    print("Device:", CFG["device"])
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # Load checkpoint
    ckpt = torch.load(CFG["model_path"], map_location="cpu")
    in_ch = ckpt["in_ch"]
    label_to_idx = ckpt["label_to_idx"]
    idx_to_label = ckpt["idx_to_label"]
    num_classes = len(label_to_idx)

    # Build model
    model = EdgeSegNet(in_ch=in_ch, num_classes=num_classes, k=CFG["k"]).to(CFG["device"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Read raw LAZ
    print("Reading:", CFG["raw_input"])
    las = laspy.read(CFG["raw_input"])
    n = len(las.points)
    print("Points:", f"{n:,}")

    xyz, feats = build_features_from_las(las, use_features=CFG["use_features"])

    # Build input X: [xyz | feats]
    if CFG["use_features"] and feats is not None:
        X = np.concatenate([xyz, feats], axis=1).astype(np.float32)
    else:
        X = xyz.astype(np.float32)

    if X.shape[1] != in_ch:
        raise RuntimeError(f"Feature mismatch: model expects {in_ch} channels, but got {X.shape[1]}")

    # Create blocks
    blocks = make_blocks_indices(xyz, CFG["block_points"])
    print("Blocks:", len(blocks))

    # Predict label indices 0..C-1
    pred_idx = run_inference(
        model=model,
        x_all=X,
        blocks=blocks,
        batch_blocks=CFG["batch_blocks"],
        device=CFG["device"]
    )

    # Map to ASPRS labels using idx_to_label
    pred_lbl = np.vectorize(lambda i: int(idx_to_label[int(i)]))(pred_idx).astype(np.int32)

    # Safety clamp to valid classes
    valid = np.array(CFG["valid_classes"], dtype=np.int32)
    bad = ~np.isin(pred_lbl, valid)
    if bad.any():
        pred_lbl[bad] = 1

    # Write output LAZ
    out = laspy.LasData(header=las.header)
    out.points = las.points
    out.classification = pred_lbl.astype(np.uint8)
    out.write(CFG["out_laz"])

    # Summary
    u, c = np.unique(pred_lbl, return_counts=True)
    print("\nSaved:", CFG["out_laz"])
    print("Class counts:")
    for cls, cnt in zip(u, c):
        print(f"  Class {int(cls):2d}: {int(cnt):,}")

if __name__ == "__main__":
    main()