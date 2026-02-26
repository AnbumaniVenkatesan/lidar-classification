import os, glob, argparse
import numpy as np
import laspy
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.neighbors import KDTree
from sklearn.metrics import classification_report, confusion_matrix
from torch_geometric.nn import DynamicEdgeConv, MLP


# ----------------------------
# Model definition (must match training)
# ----------------------------
class EdgeSegNet(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, k: int = 16):
        super().__init__()
        self.ec1 = DynamicEdgeConv(MLP([2 * in_ch, 64, 64, 64], norm="batch_norm"), k=k, aggr="max")
        self.ec2 = DynamicEdgeConv(MLP([2 * 64, 128, 128], norm="batch_norm"), k=k, aggr="max")
        self.ec3 = DynamicEdgeConv(MLP([2 * 128, 256, 256], norm="batch_norm"), k=k, aggr="max")
        # Keep head simple and stable (indices 0..4)
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


def load_edgeconv_checkpoint(model: nn.Module, ckpt: dict):
    """
    Loads ckpt saved either as:
      - ckpt["model_state"]  (your old scripts)
      - ckpt["model"]
    And fixes head index mismatch (head.3/head.5 -> head.2/head.4) if needed.
    """
    sd = ckpt.get("model_state", ckpt.get("model", None))
    if sd is None:
        raise RuntimeError("Checkpoint missing 'model_state' or 'model'.")

    keys = set(sd.keys())
    # If checkpoint has head.3/head.5 (extra modules), remap to our head.2/head.4
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
    print("✅ Loaded checkpoint.")
    if missing:
        print("   Missing keys ignored:", missing[:8], "..." if len(missing) > 8 else "")
    if unexpected:
        print("   Unexpected keys ignored:", unexpected[:8], "..." if len(unexpected) > 8 else "")


# ----------------------------
# LAZ features + blocks
# ----------------------------
def build_features_from_las(las: laspy.LasData):
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


def sample_block_indices_xy(xyz: np.ndarray, center_idx: int, block_points: int):
    cxy = xyz[center_idx, :2]
    d2 = np.sum((xyz[:, :2] - cxy[None, :]) ** 2, axis=1)
    idx = np.argpartition(d2, block_points)[:block_points]
    return idx.astype(np.int64)


def make_blocks_indices(xyz: np.ndarray, block_points: int, overlap_div: int = 4):
    """
    overlap_div=4 -> step = block_points//4 (more votes, smoother, higher confidence)
    """
    n = len(xyz)
    if n <= block_points:
        return [np.arange(n, dtype=np.int64)]

    centers = np.random.permutation(n)
    blocks = []
    step = max(1, block_points // overlap_div)

    for i in range(0, n, step):
        blocks.append(sample_block_indices_xy(xyz, int(centers[i]), block_points))
    return blocks


def predict_labels(model, X_all, blocks, batch_blocks, device):
    model.eval()
    num_classes = model.head[-1].out_features
    N = X_all.shape[0]
    vote = np.zeros((N, num_classes), dtype=np.int32)
    softmax = torch.nn.Softmax(dim=1)

    conf_sum = np.zeros((N,), dtype=np.float32)
    conf_cnt = np.zeros((N,), dtype=np.int32)

    with torch.no_grad():
        for s in tqdm(range(0, len(blocks), batch_blocks), desc="Predict blocks"):
            batch = blocks[s:s + batch_blocks]

            xs, bs, ids = [], [], []
            for bi, idx in enumerate(batch):
                xb = X_all[idx].copy()
                # simple block normalization (like yesterday’s scripts)
                xb[:, 0] -= xb[:, 0].mean()
                xb[:, 1] -= xb[:, 1].mean()
                xb[:, 2] -= xb[:, 2].min()
                xs.append(xb)
                bs.append(np.full((len(idx),), bi, dtype=np.int64))
                ids.append(idx)

            xcat = np.concatenate(xs, axis=0).astype(np.float32)
            bcat = np.concatenate(bs, axis=0).astype(np.int64)

            xt = torch.from_numpy(xcat).to(device, non_blocking=True)
            bt = torch.from_numpy(bcat).to(device, non_blocking=True)

            logits = model(xt, bt)
            prob = softmax(logits)
            pred = prob.argmax(1).cpu().numpy()
            maxp = prob.max(1).values.cpu().numpy()

            off = 0
            for idx in ids:
                p = pred[off:off + len(idx)]
                m = maxp[off:off + len(idx)]
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


def save_pred(in_las, out_path, labels):
    out = laspy.LasData(header=in_las.header)
    out.points = in_las.points
    out.classification = labels.astype(np.uint8)
    out.write(out_path)


# ----------------------------
# REAL accuracy (only if GT for SAME AREA exists)
# ----------------------------
def real_accuracy_spatial(gt_path, pred_path, classes=None):
    gt = laspy.read(gt_path)
    pr = laspy.read(pred_path)

    gxyz = np.vstack([gt.x, gt.y, gt.z]).T.astype(np.float32)
    pxyz = np.vstack([pr.x, pr.y, pr.z]).T.astype(np.float32)
    gcls = np.array(gt.classification, dtype=np.int32)
    pcls = np.array(pr.classification, dtype=np.int32)

    # Match PRED points to nearest GT points
    tree = KDTree(gxyz)
    _, idx = tree.query(pxyz, k=1)
    gmatch = gcls[idx[:, 0]]

    if classes is None:
        classes = sorted(list(set(np.unique(gmatch)).union(set(np.unique(pcls)))))

    m = np.isin(gmatch, classes)
    y_true = gmatch[m]
    y_pred = pcls[m]

    rep = classification_report(y_true, y_pred, labels=classes, digits=4, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    overall = (np.trace(cm) / np.sum(cm)) * 100.0 if cm.sum() > 0 else 0.0

    per_class_recall = {c: rep[str(c)]["recall"] * 100.0 for c in classes if str(c) in rep}
    return overall, per_class_recall


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to edgeconv_best.pth")
    ap.add_argument("--input", required=True, help="LAZ file OR folder of .laz")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--gt", default="", help="(optional) GT folder with same filenames for real accuracy")
    ap.add_argument("--block_points", type=int, default=4096)
    ap.add_argument("--batch_blocks", type=int, default=3)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--overlap_div", type=int, default=4, help="4 = good quality, 8 = better but slower")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    os.makedirs(args.out, exist_ok=True)

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)

    # try to read metadata from ckpt, else use safe defaults
    in_ch = ckpt.get("in_ch", 6)  # xyz + 3 features
    label_to_idx = ckpt.get("label_to_idx", None)
    idx_to_label = ckpt.get("idx_to_label", None)
    if label_to_idx is None or idx_to_label is None:
        num_classes = ckpt.get("num_classes", 6)
        idx_to_label = {i: i for i in range(num_classes)}
    else:
        num_classes = len(label_to_idx)

    model = EdgeSegNet(in_ch=in_ch, num_classes=num_classes, k=args.k).to(device)
    load_edgeconv_checkpoint(model, ckpt)
    model.eval()

    # Collect files
    if os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, "*.laz")))
    else:
        files = [args.input]

    if not files:
        raise RuntimeError("No .laz files found.")

    # Classes we usually report (edit as needed)
    report_classes = [1, 2, 3, 6, 7, 9, 10, 12, 13]

    for f in files:
        name = os.path.basename(f)
        base = os.path.splitext(name)[0]
        out_path = os.path.join(args.out, base + "_PRED.laz")

        print("\n" + "=" * 70)
        print("RAW:", f)
        las = laspy.read(f)

        xyz, X = build_features_from_las(las)
        if X.shape[1] != in_ch:
            raise RuntimeError(f"Feature mismatch: model expects {in_ch}, got {X.shape[1]}")

        blocks = make_blocks_indices(xyz, args.block_points, overlap_div=args.overlap_div)
        print(f"Points: {len(xyz):,} | Blocks: {len(blocks)}")

        pred_idx, conf = predict_labels(model, X, blocks, args.batch_blocks, device)
        pred_lbl = np.vectorize(lambda i: int(idx_to_label[int(i)]))(pred_idx).astype(np.int32)

        save_pred(las, out_path, pred_lbl)
        print("Saved:", out_path)

        # Distribution + confidence summary
        u, c = np.unique(pred_lbl, return_counts=True)
        total = len(pred_lbl)
        print("Class distribution:")
        for a, b in zip(u, c):
            print(f"  Class {int(a):2d}: {int(b):,} ({b/total*100:.2f}%)")
        print(f"Avg confidence: {conf.mean():.3f}")

        # Optional REAL accuracy (only if GT exists for same tile name)
        if args.gt and os.path.isdir(args.gt):
            gt_path = os.path.join(args.gt, name)  # must match filename
            if os.path.exists(gt_path):
                overall, per_cls = real_accuracy_spatial(gt_path, out_path, classes=report_classes)
                print("\nREAL ACCURACY (GT vs PRED):")
                for k in [2, 3, 6]:
                    if k in per_cls:
                        print(f"  Class {k} Recall (accuracy): {per_cls[k]:.2f}%")
                print(f"  Overall accuracy: {overall:.2f}%")
            else:
                print("GT not found for:", name)

if __name__ == "__main__":
    main()