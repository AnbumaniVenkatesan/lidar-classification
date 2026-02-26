# ============================================================
#   RandLA-Net STYLE (PyTorch) — STABLE VERSION (NO PyG knn/fps)
#   ✅ Works with your dataset .npz keys: ['points','labels','features']
#   ✅ Fixes CUDA "index out of bounds" by using LOCAL KNN via torch.cdist
#   ✅ Simple downsample (random) instead of FPS (stable)
#
#   Train:  D:\lidarrrrr\anbu\randla_dataset\train
#   Val:    D:\lidarrrrr\anbu\randla_dataset\val
#   Predict: one raw .laz -> *_PRED_RANDLA.laz
# ============================================================

import os, glob, math
import numpy as np
import laspy, lazrs
import torch
import torch.nn as nn
from tqdm import tqdm

# ============================================================
# CONFIG (EDIT PATHS)
# ============================================================
CFG = {
    "train_blocks_dir": r"D:\lidarrrrr\anbu\randla_dataset\train",
    "val_blocks_dir":   r"D:\lidarrrrr\anbu\randla_dataset\val",

    "predict_input": r"D:\lidarrrrr\anbu\INPUT FILE\DX3013595 PASQUILIO\LAZ\DX3013595 PASQUILIO000001.laz",
    "predict_out":   r"d:\lidarrrrr\anbu\test\RAW_000001_PRED_RANDLA.laz",

    "num_points": 4096,
    "k": 16,
    "epochs": 60,
    "batch_size": 6,      # RTX 3050 safe (adjust if OOM)
    "lr": 1e-3,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # Scope classes you want
    "label_set": [1, 2, 3, 6],  # Default, Ground, Veg, Building

    # ✅ remap (edit if needed). You requested: class 5 -> building 6
    "train_label_remap": {
        0: 1, 1: 1,
        2: 2,
        3: 3, 4: 3,
        5: 6, 6: 6,
        12: 1, 14: 1, 16: 1, 17: 1,
        19: 3, 21: 3, 22: 3
    }
}

# ============================================================
# DATASET
# ============================================================
class BlocksNPZ(torch.utils.data.Dataset):
    def __init__(self, folder, npts=4096, require_y=True, label_remap=None):
        self.files = sorted(glob.glob(os.path.join(folder, "*.npz")))
        if len(self.files) == 0:
            raise RuntimeError(f"No .npz blocks found in: {folder}")
        self.npts = npts
        self.require_y = require_y
        self.label_remap = label_remap or {}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])

        # ✅ your format
        pts  = d["points"].astype(np.float32)      # (N,>=3)
        xyz  = pts[:, :3].astype(np.float32)       # (N,3)
        feat = d["features"].astype(np.float32)    # (N,C)

        y = None
        if self.require_y:
            y = d["labels"].astype(np.int64)
            if self.label_remap:
                yy = y.copy()
                for old, new in self.label_remap.items():
                    yy[yy == old] = new
                y = yy

        # sample/pad
        N = xyz.shape[0]
        if N >= self.npts:
            idx = np.random.choice(N, self.npts, replace=False)
        else:
            idx = np.random.choice(N, self.npts, replace=True)

        xyz = xyz[idx]
        feat = feat[idx]
        if y is not None:
            y = y[idx]

        # normalize xyz per block
        xyz[:, 0] -= xyz[:, 0].mean()
        xyz[:, 1] -= xyz[:, 1].mean()
        xyz[:, 2] -= xyz[:, 2].min()

        # keep xyz consistent if feature begins with xyz
        if feat.ndim == 2 and feat.shape[1] >= 3:
            feat[:, 0:3] = xyz

        return xyz, feat, y

def collate_fn(batch):
    xyz  = torch.from_numpy(np.stack([b[0] for b in batch], 0)).float()  # (B,N,3)
    feat = torch.from_numpy(np.stack([b[1] for b in batch], 0)).float()  # (B,N,C)
    y0 = batch[0][2]
    y = None if y0 is None else torch.from_numpy(np.stack([b[2] for b in batch], 0)).long()
    return xyz, feat, y

# ============================================================
# KNN + Downsample helpers (LOCAL indices, stable)
# ============================================================
def knn_indices_local(xyz, k):
    """
    xyz: (B,N,3)
    returns idx: (B,N,k) with LOCAL indices 0..N-1
    """
    # (B,N,N)
    d = torch.cdist(xyz, xyz)
    idx = d.topk(k=k, dim=-1, largest=False).indices  # (B,N,k)
    return idx

def random_downsample(xyz, feat, ratio):
    """
    xyz: (B,N,3), feat: (B,C,N)
    returns xyz_ds: (B,M,3), feat_ds: (B,C,M)
    """
    B, N, _ = xyz.shape
    M = max(32, int(math.ceil(N * ratio)))
    out_xyz, out_feat = [], []
    for b in range(B):
        sel = torch.randint(0, N, (M,), device=xyz.device)
        out_xyz.append(xyz[b, sel])
        out_feat.append(feat[b, :, sel])
    return torch.stack(out_xyz, 0), torch.stack(out_feat, 0)

def nearest_interp(src_xyz, src_feat, tgt_xyz):
    """
    src_xyz: (B,M,3), src_feat: (B,C,M)
    tgt_xyz: (B,N,3)
    returns:  (B,C,N)
    """
    d = torch.cdist(tgt_xyz, src_xyz)   # (B,N,M)
    nn = d.argmin(dim=-1)               # (B,N) local 0..M-1
    B, C, M = src_feat.shape
    N = tgt_xyz.shape[1]
    idxf = nn.unsqueeze(1).expand(B, C, N)
    up = torch.gather(src_feat, 2, idxf)
    return up

# ============================================================
# RandLA-style blocks
# ============================================================
class SharedMLP1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class AttentivePool(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1, bias=False),
            nn.Softmax(dim=3)
        )
        self.mlp = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )
    def forward(self, x):  # (B,C,N,K)
        a = self.score(x)                       # (B,C,N,K)
        x = (x * a).sum(dim=3, keepdim=True)    # (B,C,N,1)
        x = self.mlp(x).squeeze(3)              # (B,out,N)
        return x

def relative_pos_encoding(xyz, neigh_xyz):
    # xyz: (B,N,3), neigh_xyz: (B,N,K,3)
    B, N, K, _ = neigh_xyz.shape
    center = xyz.unsqueeze(2).expand(B, N, K, 3)
    diff = center - neigh_xyz
    dist = torch.norm(diff, dim=3, keepdim=True)
    # (B,N,K,10)
    return torch.cat([diff, dist, center, neigh_xyz], dim=3)

class LFA(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv2d(in_ch + 10, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.pool = AttentivePool(out_ch, out_ch)

    def forward(self, xyz, feat, neigh_idx):
        """
        xyz: (B,N,3)
        feat: (B,C,N)
        neigh_idx: (B,N,K) LOCAL indices
        """
        B, C, N = feat.shape
        K = neigh_idx.shape[2]

        # gather neighbor features
        idxf = neigh_idx.unsqueeze(1).expand(B, C, N, K)  # (B,C,N,K)
        neigh_feat = torch.gather(feat.unsqueeze(3).expand(B, C, N, K), 2, idxf)

        # gather neighbor xyz
        xyz_t = xyz.permute(0, 2, 1)  # (B,3,N)
        idxxyz = neigh_idx.unsqueeze(1).expand(B, 3, N, K)
        neigh_xyz = torch.gather(xyz_t.unsqueeze(3).expand(B, 3, N, K), 2, idxxyz)
        neigh_xyz = neigh_xyz.permute(0, 2, 3, 1)  # (B,N,K,3)

        rel = relative_pos_encoding(xyz, neigh_xyz).permute(0, 3, 1, 2)  # (B,10,N,K)

        x = torch.cat([neigh_feat, rel], dim=1)  # (B,C+10,N,K)
        x = self.mlp1(x)
        x = self.pool(x)                         # (B,out,N)
        return x

# ============================================================
# RandLA-Net style (small, stable)
# ============================================================
class RandLANet(nn.Module):
    def __init__(self, in_feat, num_classes, k=16):
        super().__init__()
        self.k = k
        self.pre = SharedMLP1D(in_feat, 64)

        self.lfa1 = LFA(64, 64)
        self.lfa2 = LFA(64, 128)
        self.lfa3 = LFA(128, 256)

        self.up2 = SharedMLP1D(256 + 128, 128)
        self.up1 = SharedMLP1D(128 + 64, 64)

        self.head = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.3),
            nn.Conv1d(64, num_classes, 1)
        )

    def forward(self, xyz, feat):
        # xyz: (B,N,3), feat: (B,N,C)
        feat = feat.permute(0, 2, 1)  # (B,C,N)
        x0 = self.pre(feat)           # (B,64,N)

        n0 = knn_indices_local(xyz, self.k)      # (B,N,K)
        x1 = self.lfa1(xyz, x0, n0)              # (B,64,N)

        xyz1, f1 = random_downsample(xyz, x1, 0.5)   # (B,N/2,3), (B,64,N/2)
        n1 = knn_indices_local(xyz1, self.k)
        x2 = self.lfa2(xyz1, f1, n1)              # (B,128,N/2)

        xyz2, f2 = random_downsample(xyz1, x2, 0.5)  # (B,N/4,3), (B,128,N/4)
        n2 = knn_indices_local(xyz2, self.k)
        x3 = self.lfa3(xyz2, f2, n2)              # (B,256,N/4)

        up2 = nearest_interp(xyz2, x3, xyz1)      # (B,256,N/2)
        d2 = self.up2(torch.cat([up2, x2], dim=1)) # (B,128,N/2)

        up1 = nearest_interp(xyz1, d2, xyz)       # (B,128,N)
        d1 = self.up1(torch.cat([up1, x1], dim=1)) # (B,64,N)

        logits = self.head(d1)                    # (B,K,N)
        return logits.permute(0, 2, 1)            # (B,N,K)

# ============================================================
# TRAIN
# ============================================================
def compute_class_weights(ds, label_map, sample_blocks=200):
    counts = torch.zeros(len(label_map), dtype=torch.float64)
    n = min(len(ds), sample_blocks)
    for i in range(n):
        _, _, y = ds[i]
        for c, idx in label_map.items():
            counts[idx] += (y == c).sum()
    counts = counts + 1.0
    w = counts.sum() / counts
    w = (w / w.mean()).float()
    return w

def train():
    device = CFG["device"]
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    train_ds = BlocksNPZ(CFG["train_blocks_dir"], npts=CFG["num_points"], require_y=True,
                         label_remap=CFG.get("train_label_remap", None))
    val_ds   = BlocksNPZ(CFG["val_blocks_dir"],   npts=CFG["num_points"], require_y=True,
                         label_remap=CFG.get("train_label_remap", None))

    train_ld = torch.utils.data.DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                                           num_workers=0, collate_fn=collate_fn, drop_last=True)
    val_ld   = torch.utils.data.DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False,
                                           num_workers=0, collate_fn=collate_fn, drop_last=False)

    classes = CFG["label_set"]
    label_map = {c: i for i, c in enumerate(classes)}
    inv_map   = {i: c for c, i in label_map.items()}
    num_classes = len(classes)

    # infer in_feat
    _, feat0, _ = train_ds[0]
    in_feat = feat0.shape[1]
    print("Input feature dim:", in_feat)

    model = RandLANet(in_feat=in_feat, num_classes=num_classes, k=CFG["k"]).to(device)

    w = compute_class_weights(train_ds, label_map).to(device)
    print("Class weights:", w.detach().cpu().numpy())

    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"])
    loss_fn = nn.CrossEntropyLoss(weight=w)

    out_dir = os.path.dirname(CFG["predict_out"])
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "randla_best.pth")

    best = 0.0
    for ep in range(1, CFG["epochs"] + 1):
        model.train()
        tot, corr, loss_sum, steps = 0, 0, 0.0, 0

        for xyz, feat, y in tqdm(train_ld, desc=f"Train {ep}/{CFG['epochs']}", leave=False):
            xyz = xyz.to(device, non_blocking=True)
            feat = feat.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # map y -> 0..K-1
            y_m = torch.full_like(y, -1)
            for c, idx in label_map.items():
                y_m[y == c] = idx
            mask = (y_m >= 0)
            if mask.sum() == 0:
                continue

            opt.zero_grad(set_to_none=True)
            logits = model(xyz, feat)  # (B,N,K)

            loss = loss_fn(logits[mask], y_m[mask])
            loss.backward()
            opt.step()

            pred = logits.argmax(-1)
            corr += (pred[mask] == y_m[mask]).sum().item()
            tot  += mask.sum().item()
            loss_sum += float(loss.item())
            steps += 1

        sch.step()
        train_acc = 100.0 * corr / max(1, tot)
        avg_loss = loss_sum / max(1, steps)

        # val
        model.eval()
        v_tot, v_corr = 0, 0
        with torch.no_grad():
            for xyz, feat, y in val_ld:
                xyz = xyz.to(device, non_blocking=True)
                feat = feat.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                y_m = torch.full_like(y, -1)
                for c, idx in label_map.items():
                    y_m[y == c] = idx
                mask = (y_m >= 0)
                if mask.sum() == 0:
                    continue

                logits = model(xyz, feat)
                pred = logits.argmax(-1)
                v_corr += (pred[mask] == y_m[mask]).sum().item()
                v_tot  += mask.sum().item()

        val_acc = 100.0 * v_corr / max(1, v_tot)

        if val_acc > best:
            best = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "in_feat": in_feat,
                "classes": classes,
                "label_map": label_map,
                "inv_map": inv_map
            }, ckpt_path)

        if ep == 1 or ep % 5 == 0:
            gpu = ""
            if torch.cuda.is_available():
                gpu = f" | GPUmem {torch.cuda.memory_allocated(0)/1024**3:.2f}GB"
            print(f"Epoch {ep:03d} | loss {avg_loss:.4f} | train {train_acc:.2f}% | val {val_acc:.2f}% | best {best:.2f}%{gpu}")

    print("✅ Training done. Best val:", best)
    print("✅ Saved:", ckpt_path)
    return ckpt_path

# ============================================================
# PREDICT (single LAZ)
# ============================================================
def predict_one(ckpt_path):
    device = CFG["device"]
    ckpt = torch.load(ckpt_path, map_location="cpu")
    inv_map = ckpt["inv_map"]
    in_feat = ckpt["in_feat"]
    classes = ckpt["classes"]

    model = RandLANet(in_feat=in_feat, num_classes=len(classes), k=CFG["k"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    las = laspy.read(CFG["predict_input"])
    dims = set(las.point_format.dimension_names)

    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)

    # Build simple features if model expects more than 3
    intensity = np.array(las.intensity, dtype=np.float32) if "intensity" in dims else np.zeros(len(xyz), np.float32)
    rn = np.array(las.return_number, dtype=np.float32) if "return_number" in dims else np.ones(len(xyz), np.float32)
    nr = np.array(las.number_of_returns, dtype=np.float32) if "number_of_returns" in dims else np.ones(len(xyz), np.float32)

    if intensity.max() > intensity.min():
        intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-6)
    ret_ratio = rn / (nr + 1e-6)

    base_feat = np.column_stack([xyz, intensity, ret_ratio, nr]).astype(np.float32)  # (N,6)

    if base_feat.shape[1] > in_feat:
        feat = base_feat[:, :in_feat]
    elif base_feat.shape[1] < in_feat:
        pad = np.zeros((len(base_feat), in_feat - base_feat.shape[1]), np.float32)
        feat = np.hstack([base_feat, pad])
    else:
        feat = base_feat

    N = len(xyz)
    NPTS = CFG["num_points"]
    bs = CFG["batch_size"]

    order = np.arange(N)
    np.random.shuffle(order)

    blocks = []
    for i in range(0, N, NPTS):
        b = order[i:i+NPTS]
        if len(b) < NPTS:
            b = np.pad(b, (0, NPTS - len(b)), mode="wrap")
        blocks.append(b)

    pred_out = np.zeros(N, dtype=np.int32)

    with torch.no_grad():
        for i in tqdm(range(0, len(blocks), bs), desc="Predict"):
            chunk = blocks[i:i+bs]
            B = len(chunk)

            bx = np.stack([xyz[b] for b in chunk], 0).astype(np.float32)   # (B,N,3)
            bf = np.stack([feat[b] for b in chunk], 0).astype(np.float32)  # (B,N,C)

            # normalize xyz in each block
            bx[:, :, 0] -= bx[:, :, 0].mean(axis=1, keepdims=True)
            bx[:, :, 1] -= bx[:, :, 1].mean(axis=1, keepdims=True)
            bx[:, :, 2] -= bx[:, :, 2].min(axis=1, keepdims=True)

            if bf.shape[2] >= 3:
                bf[:, :, 0:3] = bx

            tx = torch.from_numpy(bx).float().to(device)
            tf = torch.from_numpy(bf).float().to(device)

            logits = model(tx, tf)                 # (B,N,K)
            p = logits.argmax(-1).cpu().numpy()    # (B,N)

            for bi in range(B):
                mapped = np.vectorize(inv_map.get)(p[bi])
                pred_out[chunk[bi]] = mapped.astype(np.int32)

    out = laspy.LasData(header=las.header)
    out.points = las.points
    out.classification = pred_out.astype(np.uint8)
    os.makedirs(os.path.dirname(CFG["predict_out"]), exist_ok=True)
    out.write(CFG["predict_out"])

    u, c = np.unique(pred_out, return_counts=True)
    print("✅ Saved:", CFG["predict_out"])
    print("Class counts:", dict(zip(u.tolist(), c.tolist())))

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=== RandLA-Net style (PyTorch) ===")
    print("Train blocks dir:", CFG["train_blocks_dir"])
    print("Val blocks dir  :", CFG["val_blocks_dir"])
    print("Device:", CFG["device"])
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    ckpt = train()
    predict_one(ckpt)