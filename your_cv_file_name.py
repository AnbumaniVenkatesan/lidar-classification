"""
Leave-One-Out Cross Validation (LOOCV) for LiDAR classified .LAZ files
Real (neighbor-based) RandLA-Net-Lite style block model
Classes trained/evaluated: 2,3,4,5,6  (others ignored for accuracy; predicted -> other_class if low confidence)

✅ Fixes included:
- SAFE index_points() (no NxN memory blow-up)
- SAFE label mapping (no np.vectorize on empty)
- tqdm progress per epoch
- New PyTorch AMP API (torch.amp.autocast / GradScaler) so warnings stop
- Auto-load all .laz files from folder using glob
- GPU-friendly defaults for RTX 3050 6GB
"""

import os
import glob
import numpy as np
import laspy
from tqdm import tqdm
from sklearn.neighbors import KDTree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import torch
import torch.nn as nn

# ============================================================
# CONFIG  ✅ EDIT THIS
# ============================================================

CONFIG = {
    # Folder containing your 10 classified LAZ files
    "folder": r"d:\lidarrrrr\anbu\training_labeled",

    # Output directory for predicted folds
    "out_dir": r"d:\lidarrrrr\anbu\cv_outputs",

    # Train/evaluate only these classes
    "target_classes": [2, 3, 4, 5, 6],
    "other_class": 1,      # used only if conf_thresh > 0

    # Block sampling / neighbors (GPU safe)
    "tile_size": 50.0,     # meters (XY tile)
    "n_points": 1024,      # 1024 safe. Try 2048 later.
    "k": 8,                # 8 safe. Try 16 later.

    # Training (per fold)
    "epochs": 10,          # start small; try 20 later
    "steps_per_epoch": 200, # start small; try 400/600 later
    "batch_size": 2,       # RTX 3050 safe. If OOM -> 1

    "lr": 1e-3,
    "weight_decay": 1e-4,

    # Prediction
    "pred_block": 2048,    # chunk size for prediction within tile
    "conf_thresh": 0.0,    # set 0.55 to map low-confidence -> other_class

    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# ============================================================
# Build file list automatically
# ============================================================

FILES = sorted(glob.glob(os.path.join(CONFIG["folder"], "*.laz")))
print("Total files found:", len(FILES))
for p in FILES:
    print(" -", p)

if len(FILES) < 2:
    raise SystemExit("❌ Need at least 2 .laz files for cross validation.")

os.makedirs(CONFIG["out_dir"], exist_ok=True)

# ============================================================
# DATA INDEX (tile-based sampling)
# ============================================================

class LidarFileIndex:
    def __init__(self, path, tile_size: float):
        self.path = path
        self.tile_size = float(tile_size)

        las = laspy.read(path)
        dims = list(las.point_format.dimension_names)

        self.x = np.asarray(las.x, dtype=np.float32)
        self.y = np.asarray(las.y, dtype=np.float32)
        self.z = np.asarray(las.z, dtype=np.float32)
        self.n = len(self.x)

        self.intensity = np.asarray(las.intensity, dtype=np.float32) if "intensity" in dims else np.zeros(self.n, np.float32)
        self.return_num = np.asarray(las.return_number, dtype=np.float32) if "return_number" in dims else np.ones(self.n, np.float32)
        self.num_returns = np.asarray(las.number_of_returns, dtype=np.float32) if "number_of_returns" in dims else np.ones(self.n, np.float32)

        self.cls = np.asarray(las.classification, dtype=np.int32)

        xmin, ymin = float(self.x.min()), float(self.y.min())
        ix = np.floor((self.x - xmin) / self.tile_size).astype(np.int32)
        iy = np.floor((self.y - ymin) / self.tile_size).astype(np.int32)

        key = (ix.astype(np.int64) << 32) ^ (iy.astype(np.int64) & 0xFFFFFFFF)

        order = np.argsort(key)
        key_sorted = key[order]
        uniq, start, counts = np.unique(key_sorted, return_index=True, return_counts=True)

        self.order = order
        self.tile_keys = uniq
        self.tile_start = start
        self.tile_counts = counts

    def sample_block_indices(self, n_points: int):
        t = np.random.randint(0, len(self.tile_keys))
        s = int(self.tile_start[t])
        c = int(self.tile_counts[t])
        idx = self.order[s:s + c]

        if c >= n_points:
            pick = np.random.choice(idx, n_points, replace=False)
        else:
            pick = np.random.choice(idx, n_points, replace=True)
        return pick

def build_block(f: LidarFileIndex, idx: np.ndarray, target_set: set, cfg: dict):
    # Points
    pts = np.stack([f.x[idx], f.y[idx], f.z[idx]], axis=1).astype(np.float32)

    # Normalize local coordinates
    center = pts.mean(axis=0, keepdims=True)
    pts0 = pts - center
    scale = np.max(np.linalg.norm(pts0[:, :2], axis=1)) + 1e-6
    ptsn = pts0 / scale

    # Simple input features (RandLA-style: coords + intensity + return ratio)
    inten = f.intensity[idx]
    denom = (inten.max() - inten.min()) + 1e-6
    inten = (inten - inten.min()) / denom

    ret_ratio = f.return_num[idx] / (f.num_returns[idx] + 1e-6)

    feat = np.stack([ptsn[:, 0], ptsn[:, 1], ptsn[:, 2], inten, ret_ratio], axis=1).astype(np.float32)

    # Labels mapping: {2,3,4,5,6} -> {0,1,2,3,4}
    cls = f.cls[idx].astype(np.int32)
    labels = np.full(cls.shape, -1, dtype=np.int64)

    targets = sorted(list(target_set))
    map_to = {c: i for i, c in enumerate(targets)}

    m = np.isin(cls, targets)
    if m.any():
        labels[m] = np.array([map_to[int(c)] for c in cls[m]], dtype=np.int64)

    # kNN within block
    tree = KDTree(ptsn, leaf_size=32)
    knn = tree.query(ptsn, k=cfg["k"], return_distance=False).astype(np.int64)

    return pts.astype(np.float32), feat.astype(np.float32), labels.astype(np.int64), knn.astype(np.int64)

# ============================================================
# MODEL (RandLA-Net-Lite style neighbor aggregation)
# ============================================================

def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    SAFE gather without expanding to (B,N,N,C).

    points: (B, N, C)
    idx:    (B, N, k)
    return: (B, N, k, C)
    """
    B, N, C = points.shape
    k = idx.shape[-1]

    points_flat = points.reshape(B * N, C)  # (B*N, C)
    offset = (torch.arange(B, device=points.device) * N).view(B, 1, 1)
    idx_global = idx + offset  # (B,N,k)

    gathered = points_flat[idx_global.reshape(-1)]  # (B*N*k, C)
    return gathered.reshape(B, N, k, C)

class SharedMLP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.fc = nn.Linear(in_ch, out_ch, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        # x: (B,N,C)
        B, N, C = x.shape
        x = self.fc(x)  # (B,N,out)
        x = self.bn(x.reshape(B * N, -1)).reshape(B, N, -1)
        return self.act(x)

class AttentivePooling(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.score = nn.Linear(in_ch, 1, bias=False)
        self.mlp = SharedMLP(in_ch, out_ch)

    def forward(self, neigh_feat):
        # neigh_feat: (B,N,k,C)
        score = self.score(neigh_feat)                 # (B,N,k,1)
        attn = torch.softmax(score, dim=2)             # (B,N,k,1)
        agg = torch.sum(attn * neigh_feat, dim=2)      # (B,N,C)
        return self.mlp(agg)                           # (B,N,out)

class LocalFeatureAggregation(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mlp1 = SharedMLP(in_ch, out_ch // 2)
        self.mlp2 = SharedMLP(out_ch // 2 + 10, out_ch // 2)  # neighbor feat + pos enc
        self.pool = AttentivePooling(out_ch // 2, out_ch)
        self.short = SharedMLP(in_ch, out_ch)

    def forward(self, xyz, feat, knn_idx):
        f1 = self.mlp1(feat)  # (B,N,out/2)

        neigh_xyz = index_points(xyz, knn_idx)           # (B,N,k,3)
        center_xyz = xyz.unsqueeze(2)                    # (B,N,1,3)
        rel = neigh_xyz - center_xyz                     # (B,N,k,3)
        dist = torch.norm(rel, dim=-1, keepdim=True)     # (B,N,k,1)

        # position encoding: rel(3) + dist(1) + center(3) + neigh(3) = 10
        pe = torch.cat([rel, dist, center_xyz.expand_as(neigh_xyz), neigh_xyz], dim=-1)  # (B,N,k,10)

        neigh_f = index_points(f1, knn_idx)              # (B,N,k,out/2)
        x = torch.cat([neigh_f, pe], dim=-1)             # (B,N,k,out/2+10)

        B, N, k, C = x.shape
        x2 = self.mlp2(x.reshape(B, N * k, C)).reshape(B, N, k, -1)  # (B,N,k,out/2)

        out = self.pool(x2)                               # (B,N,out)
        return out + self.short(feat)

class RandLANetLite(nn.Module):
    def __init__(self, in_feat=5, num_classes=5, k=8):
        super().__init__()
        self.k = k
        self.lfa1 = LocalFeatureAggregation(in_feat, 64)
        self.lfa2 = LocalFeatureAggregation(64, 128)
        self.lfa3 = LocalFeatureAggregation(128, 256)
        self.head = nn.Sequential(
            nn.Linear(256, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, xyz, feat, knn):
        x = self.lfa1(xyz, feat, knn)
        x = self.lfa2(xyz, x, knn)
        x = self.lfa3(xyz, x, knn)
        B, N, C = x.shape
        logits = self.head(x.reshape(B * N, C)).reshape(B, N, -1)
        return logits

# ============================================================
# TRAIN ONE FOLD
# ============================================================

def train_one_fold(train_paths, cfg):
    device = cfg["device"]
    target_set = set(cfg["target_classes"])

    # Load training files into memory indexes
    train_files = [LidarFileIndex(p, cfg["tile_size"]) for p in train_paths]

    model = RandLANetLite(in_feat=5, num_classes=5, k=cfg["k"]).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    # AMP (new API)
    use_amp = torch.cuda.is_available() and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Class weights (light balancing)
    class_weights = torch.tensor([1.0, 1.2, 1.2, 1.2, 1.2], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)

    for epoch in range(cfg["epochs"]):
        model.train()
        pbar = tqdm(range(cfg["steps_per_epoch"]), desc=f"Train epoch {epoch+1}/{cfg['epochs']}", ncols=100)

        for _ in pbar:
            B = cfg["batch_size"]
            xyz_b, feat_b, lbl_b, knn_b = [], [], [], []

            tries = 0
            while len(xyz_b) < B and tries < B * 20:
                tries += 1
                f = train_files[np.random.randint(0, len(train_files))]
                idx = f.sample_block_indices(cfg["n_points"])
                pts, feat, lbl, knn = build_block(f, idx, target_set, cfg)

                # Skip blocks with too few labeled points (2..6)
                labeled = (lbl >= 0).sum()
                if labeled < cfg["n_points"] * 0.2:
                    continue

                xyz_b.append(pts)
                feat_b.append(feat)
                lbl_b.append(lbl)
                knn_b.append(knn)

            if len(xyz_b) < B:
                continue

            xyz = torch.from_numpy(np.stack(xyz_b)).to(device)
            feat = torch.from_numpy(np.stack(feat_b)).to(device)
            lbl = torch.from_numpy(np.stack(lbl_b)).to(device)
            knn = torch.from_numpy(np.stack(knn_b)).to(device)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(xyz, feat, knn)                  # (B,N,5)
                loss = criterion(logits.permute(0, 2, 1), lbl)  # (B,C,N)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            pbar.set_postfix(loss=float(loss.item()))

        sch.step()

    return model

# ============================================================
# PREDICT ONE FILE + SAVE PREDICTED LAZ
# ============================================================

def predict_and_save(model, test_path, out_path, cfg):
    device = cfg["device"]
    use_amp = torch.cuda.is_available() and device.startswith("cuda")
    model.eval()

    las = laspy.read(test_path)
    dims = list(las.point_format.dimension_names)

    x = np.asarray(las.x, dtype=np.float32)
    y = np.asarray(las.y, dtype=np.float32)
    z = np.asarray(las.z, dtype=np.float32)
    n = len(x)

    intensity = np.asarray(las.intensity, dtype=np.float32) if "intensity" in dims else np.zeros(n, np.float32)
    return_num = np.asarray(las.return_number, dtype=np.float32) if "return_number" in dims else np.ones(n, np.float32)
    num_returns = np.asarray(las.number_of_returns, dtype=np.float32) if "number_of_returns" in dims else np.ones(n, np.float32)

    # Tile grouping
    tile = float(cfg["tile_size"])
    xmin, ymin = float(x.min()), float(y.min())
    ix = np.floor((x - xmin) / tile).astype(np.int32)
    iy = np.floor((y - ymin) / tile).astype(np.int32)
    key = (ix.astype(np.int64) << 32) ^ (iy.astype(np.int64) & 0xFFFFFFFF)

    order = np.argsort(key)
    key_sorted = key[order]
    uniq, start, counts = np.unique(key_sorted, return_index=True, return_counts=True)

    # map model classes 0..4 -> {2,3,4,5,6}
    targets = cfg["target_classes"]
    inv = {0: targets[0], 1: targets[1], 2: targets[2], 3: targets[3], 4: targets[4]}

    out_cls = np.full(n, cfg["other_class"], dtype=np.uint8)

    for t in tqdm(range(len(uniq)), desc="Predict tiles", ncols=100):
        s = int(start[t]); c = int(counts[t])
        idx_tile = order[s:s + c]

        for s2 in range(0, c, cfg["pred_block"]):
            e2 = min(s2 + cfg["pred_block"], c)
            idx = idx_tile[s2:e2]

            pts = np.stack([x[idx], y[idx], z[idx]], axis=1).astype(np.float32)
            center = pts.mean(axis=0, keepdims=True)
            pts0 = pts - center
            scale = np.max(np.linalg.norm(pts0[:, :2], axis=1)) + 1e-6
            ptsn = pts0 / scale

            inten = intensity[idx]
            inten = (inten - inten.min()) / ((inten.max() - inten.min()) + 1e-6)
            ret_ratio = return_num[idx] / (num_returns[idx] + 1e-6)

            feat = np.stack([ptsn[:, 0], ptsn[:, 1], ptsn[:, 2], inten, ret_ratio], axis=1).astype(np.float32)

            tree = KDTree(ptsn, leaf_size=32)
            knn = tree.query(ptsn, k=cfg["k"], return_distance=False).astype(np.int64)

            xyz_t = torch.from_numpy(pts).unsqueeze(0).to(device)
            feat_t = torch.from_numpy(feat).unsqueeze(0).to(device)
            knn_t = torch.from_numpy(knn).unsqueeze(0).to(device)

            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(xyz_t, feat_t, knn_t)[0]  # (N,5)
                prob = torch.softmax(logits, dim=-1)
                conf, pred = torch.max(prob, dim=-1)

            pred = pred.cpu().numpy().astype(np.int32)
            conf = conf.cpu().numpy().astype(np.float32)

            mapped = np.array([inv[int(p)] for p in pred], dtype=np.uint8)

            if cfg["conf_thresh"] > 0:
                mapped[conf < float(cfg["conf_thresh"])] = np.uint8(cfg["other_class"])

            out_cls[idx] = mapped

    out = laspy.LasData(las.header)
    out.points = las.points
    out.classification = out_cls
    out.write(out_path)

    return out_cls

# ============================================================
# EVALUATE ONE FOLD
# ============================================================

def evaluate_fold(true_path, pred_cls, cfg):
    las_true = laspy.read(true_path)
    y_true = np.asarray(las_true.classification, dtype=np.int32)
    y_pred = pred_cls.astype(np.int32)

    mask = np.isin(y_true, cfg["target_classes"])
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    acc = accuracy_score(y_true, y_pred)
    return acc, y_true, y_pred

# ============================================================
# MAIN: LOOCV
# ============================================================

def main():
    print("\nDevice:", CONFIG["device"].upper())
    print("Classes:", CONFIG["target_classes"])
    print("Folds:", len(FILES))
    print("Outputs:", CONFIG["out_dir"])
    print("=" * 70)

    fold_accs = []

    for i in range(len(FILES)):
        test_path = FILES[i]
        train_paths = [FILES[j] for j in range(len(FILES)) if j != i]

        print(f"\nFOLD {i+1}/{len(FILES)}")
        print("Test :", os.path.basename(test_path))
        print("Train:", len(train_paths), "files")

        # Train
        model = train_one_fold(train_paths, CONFIG)

        # Predict + Save
        out_pred_path = os.path.join(
            CONFIG["out_dir"],
            f"{os.path.splitext(os.path.basename(test_path))[0]}_pred.laz"
        )
        pred_cls = predict_and_save(model, test_path, out_pred_path, CONFIG)

        # Evaluate
        acc, y_true, y_pred = evaluate_fold(test_path, pred_cls, CONFIG)
        fold_accs.append(acc)

        print(f"Fold Accuracy: {acc*100:.2f}%")
        # Uncomment if you want details each fold:
        # print(classification_report(y_true, y_pred, digits=4))
        # print("Confusion:\n", confusion_matrix(y_true, y_pred))

        # Free VRAM between folds
        del model
        torch.cuda.empty_cache()

    mean_acc = float(np.mean(fold_accs))
    std_acc = float(np.std(fold_accs))

    print("\n" + "=" * 70)
    print("✅ CROSS VALIDATION RESULTS (Leave-One-Out)")
    for i, a in enumerate(fold_accs, 1):
        print(f"Fold {i:02d}: {a*100:.2f}%")
    print("-" * 70)
    print(f"✅ Mean Accuracy: {mean_acc*100:.2f}%")
    print(f"✅ Std  Accuracy: {std_acc*100:.2f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()