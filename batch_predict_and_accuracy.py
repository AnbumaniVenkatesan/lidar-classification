import os, glob
import numpy as np
import laspy
import torch
import torch.nn as nn
from tqdm import tqdm
from torch_geometric.nn import DynamicEdgeConv, MLP

RAW_DIR   = r"d:\lidarrrrr\anbu\LAZ\LAZ"
OUT_DIR   = r"d:\lidarrrrr\anbu\out10"
MODEL_PATH = r"d:\lidarrrrr\anbu\dl_models\edgeconv_best.pth"

BLOCK_POINTS = 4096
BATCH_BLOCKS = 3
K_NEIGHBORS  = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_CLASS = 1
GROUND_CLASS  = 2
VEG_CLASS     = 3
HAG_VEG_MIN_M = 1.0
LOW_CONF_TH   = 0.55
VALID_FINAL_CLASSES = [1,2,3,6,7,9,10,12,13]

class EdgeSegNet(nn.Module):
    def __init__(self, in_ch, num_classes, k=16):
        super().__init__()
        self.ec1 = DynamicEdgeConv(MLP([2 * in_ch, 64, 64, 64], norm="batch_norm"), k=k, aggr="max")
        self.ec2 = DynamicEdgeConv(MLP([2 * 64, 128, 128], norm="batch_norm"), k=k, aggr="max")
        self.ec3 = DynamicEdgeConv(MLP([2 * 128, 256, 256], norm="batch_norm"), k=k, aggr="max")
        # IMPORTANT: keep head modules as Linear/ReLU/Linear/ReLU/Linear (indices 0..4)
        self.head = nn.Sequential(
            nn.Linear(64 + 128 + 256, 256),  # head.0
            nn.ReLU(inplace=True),           # head.1
            nn.Linear(256, 128),             # head.2
            nn.ReLU(inplace=True),           # head.3
            nn.Linear(128, num_classes),     # head.4
        )

    def forward(self, x, batch):
        f1 = self.ec1(x, batch)
        f2 = self.ec2(f1, batch)
        f3 = self.ec3(f2, batch)
        feat = torch.cat([f1, f2, f3], dim=1)
        return self.head(feat)

def load_edgeconv_checkpoint(model, ckpt):
    sd = ckpt.get("model_state", ckpt.get("model", None))
    if sd is None:
        raise RuntimeError("Checkpoint has no 'model_state' or 'model' key.")

    keys = set(sd.keys())
    # If checkpoint uses head.3/head.5 (extra modules), remap to our head.2/head.4
    if any(k.startswith("head.3.") for k in keys) and any(k.startswith("head.5.") for k in keys):
        remap = {}
        for k, v in sd.items():
            nk = k
            if nk.startswith("head.3."):
                nk = "head.2." + nk[len("head.3."):]
            elif nk.startswith("head.5."):
                nk = "head.4." + nk[len("head.5."):]
            remap[nk] = v
        sd = remap

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("Loaded checkpoint.")
    if missing:
        print("Missing keys (ignored):", missing[:8], "..." if len(missing) > 8 else "")
    if unexpected:
        print("Unexpected keys (ignored):", unexpected[:8], "..." if len(unexpected) > 8 else "")

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def build_features_from_las(las):
    dims = set(las.point_format.dimension_names)
    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)

    intensity = np.array(las.intensity, dtype=np.float32) if "intensity" in dims else np.zeros(len(xyz), np.float32)
    rn = np.array(las.return_number, dtype=np.float32) if "return_number" in dims else np.ones(len(xyz), np.float32)
    nr = np.array(las.number_of_returns, dtype=np.float32) if "number_of_returns" in dims else np.ones(len(xyz), np.float32)

    if intensity.max() > intensity.min():
        intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-6)
    ret_ratio = rn / (nr + 1e-6)

    X = np.concatenate([xyz, np.column_stack([intensity, ret_ratio, nr]).astype(np.float32)], axis=1)
    return xyz, X.astype(np.float32)

def sample_block_indices_xy(xyz, center_idx, block_points):
    cxy = xyz[center_idx, :2]
    d2 = np.sum((xyz[:, :2] - cxy[None, :]) ** 2, axis=1)
    idx = np.argpartition(d2, block_points)[:block_points]
    return idx.astype(np.int64)

def make_blocks_indices(xyz, block_points):
    n = len(xyz)
    if n <= block_points:
        return [np.arange(n, dtype=np.int64)]
    centers = np.random.permutation(n)
    blocks = []
    step = block_points
    for i in range(0, n, step):
        blocks.append(sample_block_indices_xy(xyz, int(centers[i]), block_points))
    return blocks

def apply_scope_clamp(pred_lbl):
    valid = np.array(VALID_FINAL_CLASSES, dtype=np.int32)
    bad = ~np.isin(pred_lbl, valid)
    if bad.any():
        pred_lbl[bad] = DEFAULT_CLASS
    return pred_lbl

def hag_fix_using_pred_ground(xyz, pred_lbl, grid_m=1.0):
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
    gmask = (pred_lbl == GROUND_CLASS)
    if gmask.sum() < 1000:
        return pred_lbl

    gx, gy, gz = x[gmask], y[gmask], z[gmask]
    xmin, ymin = gx.min(), gy.min()
    ix = np.floor((gx - xmin) / grid_m).astype(np.int32)
    iy = np.floor((gy - ymin) / grid_m).astype(np.int32)
    nx = int(ix.max()) + 1
    ny = int(iy.max()) + 1

    grid = np.full((nx, ny), np.inf, dtype=np.float32)
    flat = ix.astype(np.int64) * ny + iy.astype(np.int64)
    order = np.argsort(flat)
    flat_s = flat[order]
    gz_s = gz[order]
    starts = np.r_[0, np.where(flat_s[1:] != flat_s[:-1])[0] + 1]
    ends   = np.r_[starts[1:], len(flat_s)]
    for s,e in zip(starts, ends):
        fi = flat_s[s]
        cx = int(fi // ny); cy = int(fi % ny)
        grid[cx, cy] = float(np.min(gz_s[s:e]))

    gmin = np.min(grid[np.isfinite(grid)])
    grid[~np.isfinite(grid)] = gmin

    px = np.floor((x - xmin) / grid_m).astype(np.int32)
    py = np.floor((y - ymin) / grid_m).astype(np.int32)
    px = np.clip(px, 0, nx-1); py = np.clip(py, 0, ny-1)
    ground_z = grid[px, py]
    hag = z - ground_z

    out = pred_lbl.copy()
    m = (out == VEG_CLASS) & (hag < HAG_VEG_MIN_M)
    out[m] = DEFAULT_CLASS
    return out

def save_laz_like(in_las, out_path, pred_lbl):
    out = laspy.LasData(header=in_las.header)
    out.points = in_las.points
    out.classification = pred_lbl.astype(np.uint8)
    out.write(out_path)

def run_inference_with_conf(model, X_all, blocks):
    model.eval()
    num_classes = model.head[-1].out_features
    N = X_all.shape[0]
    vote = np.zeros((N, num_classes), dtype=np.int32)
    conf_sum = np.zeros((N,), dtype=np.float32)
    conf_cnt = np.zeros((N,), dtype=np.int32)
    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for s in tqdm(range(0, len(blocks), BATCH_BLOCKS), desc="Predict blocks"):
            batch = blocks[s:s+BATCH_BLOCKS]
            xs, bs, ids = [], [], []
            for bi, idx in enumerate(batch):
                xb = X_all[idx].copy()
                xb[:,0] -= xb[:,0].mean()
                xb[:,1] -= xb[:,1].mean()
                xb[:,2] -= xb[:,2].min()
                xs.append(xb)
                bs.append(np.full((len(idx),), bi, dtype=np.int64))
                ids.append(idx)

            xcat = np.concatenate(xs, axis=0).astype(np.float32)
            bcat = np.concatenate(bs, axis=0).astype(np.int64)

            xt = torch.from_numpy(xcat).to(DEVICE, non_blocking=True)
            bt = torch.from_numpy(bcat).to(DEVICE, non_blocking=True)

            logits = model(xt, bt)
            prob = softmax(logits)
            pred = prob.argmax(1).cpu().numpy()
            maxp = prob.max(1).values.cpu().numpy()

            off = 0
            for idx in ids:
                p = pred[off:off+len(idx)]
                m = maxp[off:off+len(idx)]
                vote[idx, p] += 1
                conf_sum[idx] += m
                conf_cnt[idx] += 1
                off += len(idx)

            del xt, bt, logits, prob
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    pred_idx = vote.argmax(axis=1).astype(np.int32)
    conf = conf_sum / np.maximum(conf_cnt, 1)
    return pred_idx, conf

def main():
    ensure_dir(OUT_DIR)
    print("Device:", DEVICE)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

    # These must exist in your saved ckpt. If not, we fallback.
    in_ch = ckpt.get("in_ch", 6)  # xyz + 3 features = 6
    idx_to_label = ckpt.get("idx_to_label", None)
    label_to_idx = ckpt.get("label_to_idx", None)
    if idx_to_label is None or label_to_idx is None:
        # fallback: assume classes are 0..(num_classes-1) already
        # and ckpt stores num_classes
        num_classes = ckpt.get("num_classes", 6)
        idx_to_label = {i:i for i in range(num_classes)}
    else:
        num_classes = len(label_to_idx)

    model = EdgeSegNet(in_ch=in_ch, num_classes=num_classes, k=K_NEIGHBORS).to(DEVICE)
    load_edgeconv_checkpoint(model, ckpt)
    model.eval()

    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.laz")))
    if not files:
        raise RuntimeError(f"No .laz found in {RAW_DIR}")

    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        out_path = os.path.join(OUT_DIR, name + "_PRED.laz")

        print("\n" + "="*70)
        print("RAW:", f)
        las = laspy.read(f)
        xyz, X = build_features_from_las(las)

        if X.shape[1] != in_ch:
            raise RuntimeError(f"Feature mismatch: model expects {in_ch}, got {X.shape[1]}")

        blocks = make_blocks_indices(xyz, BLOCK_POINTS)
        print("Points:", f"{len(xyz):,}", "| Blocks:", len(blocks))

        pred_idx, conf = run_inference_with_conf(model, X, blocks)
        pred_lbl = np.vectorize(lambda i: int(idx_to_label[int(i)]))(pred_idx).astype(np.int32)

        pred_lbl = hag_fix_using_pred_ground(xyz, pred_lbl, grid_m=1.0)
        pred_lbl = apply_scope_clamp(pred_lbl)

        save_laz_like(las, out_path, pred_lbl)

        u, c = np.unique(pred_lbl, return_counts=True)
        total = len(pred_lbl)
        print("Saved:", out_path)
        print("Class distribution:")
        for a, b in zip(u, c):
            print(f"  Class {int(a):2d}: {int(b):,} ({b/total*100:.2f}%)")

        low = int((conf < LOW_CONF_TH).sum())
        print(f"Avg confidence: {conf.mean():.3f} | Low-conf (<{LOW_CONF_TH}): {low:,} ({low/total*100:.2f}%)")

if __name__ == "__main__":
    main()