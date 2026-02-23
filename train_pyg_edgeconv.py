import os, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from torch_geometric.nn import DynamicEdgeConv, MLP

CFG = {
    "dataset_root": r"d:\lidarrrrr\anbu\randla_dataset",
    "epochs": 40,
    "batch_blocks": 4,              # ✅ start safe on RTX 3050 (increase later if stable)
    "num_workers": 2,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "k": 16,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_path": r"d:\lidarrrrr\anbu\dl_models\edgeconv_best.pth",
    "label_keep": [1, 2, 3, 4, 5, 6],
}

os.makedirs(os.path.dirname(CFG["save_path"]), exist_ok=True)

class BlockNPZ(Dataset):
    def __init__(self, folder):
        self.files = sorted(glob.glob(os.path.join(folder, "*.npz")))
        if len(self.files) == 0:
            raise RuntimeError(f"No npz found in: {folder}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        pts = d["points"].astype(np.float32)   # (N,3)
        lbl = d["labels"].astype(np.int64)     # (N,)
        feats = d["features"].astype(np.float32) if "features" in d.files else None

        x = pts if feats is None else np.concatenate([pts, feats], axis=1)  # (N,F)
        return torch.from_numpy(x), torch.from_numpy(lbl)

def collate_blocks(batch):
    xs, ys, batch_vec = [], [], []
    for bi, (x, y) in enumerate(batch):
        n = x.shape[0]
        xs.append(x)
        ys.append(y)
        batch_vec.append(torch.full((n,), bi, dtype=torch.long))
    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    b = torch.cat(batch_vec, dim=0)
    return x, y, b

class EdgeSegNet(nn.Module):
    def __init__(self, in_ch, num_classes, k=16):
        super().__init__()
        self.ec1 = DynamicEdgeConv(MLP([2 * in_ch, 64, 64, 64], norm="batch_norm"), k=k, aggr="max")
        self.ec2 = DynamicEdgeConv(MLP([2 * 64, 128, 128], norm="batch_norm"), k=k, aggr="max")
        self.ec3 = DynamicEdgeConv(MLP([2 * 128, 256, 256], norm="batch_norm"), k=k, aggr="max")

        self.head = nn.Sequential(
            nn.Linear(64 + 128 + 256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
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

def compute_class_weights(loader, kept_labels):
    kept_labels = list(kept_labels)
    label_to_idx = {lab: i for i, lab in enumerate(kept_labels)}
    counts = np.zeros(len(kept_labels), dtype=np.int64)

    for x, y, b in tqdm(loader, desc="Counting labels", leave=False):
        y = y.numpy()
        for lab, idx in label_to_idx.items():
            counts[idx] += np.sum(y == lab)

    counts = np.maximum(counts, 1)
    freq = counts / counts.sum()
    w = 1.0 / (freq + 1e-9)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32), label_to_idx

def main():
    torch.backends.cudnn.benchmark = True
    print("Device:", CFG["device"])
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    train_ds = BlockNPZ(os.path.join(CFG["dataset_root"], "train"))
    val_ds   = BlockNPZ(os.path.join(CFG["dataset_root"], "val"))

    train_loader = DataLoader(
        train_ds, batch_size=CFG["batch_blocks"], shuffle=True,
        num_workers=CFG["num_workers"], pin_memory=True, collate_fn=collate_blocks
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG["batch_blocks"], shuffle=False,
        num_workers=CFG["num_workers"], pin_memory=True, collate_fn=collate_blocks
    )

    # input channels
    x0, y0 = train_ds[0]
    in_ch = x0.shape[1]
    kept = CFG["label_keep"]
    num_classes = len(kept)

    # class weights
    w, label_to_idx = compute_class_weights(train_loader, kept)
    w = w.to(CFG["device"])
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    print("Class weights:", w.detach().cpu().numpy().round(3).tolist())

    model = EdgeSegNet(in_ch, num_classes, k=CFG["k"]).to(CFG["device"])
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val = 0.0

    for ep in range(1, CFG["epochs"] + 1):
        # ---- train ----
        model.train()
        tot_loss, tot_correct, tot_n = 0.0, 0, 0

        for x, y, batch in tqdm(train_loader, desc=f"Train {ep}/{CFG['epochs']}", leave=False):
            x = x.to(CFG["device"], non_blocking=True)
            y = y.to(CFG["device"], non_blocking=True)
            batch = batch.to(CFG["device"], non_blocking=True)

            # map labels to [0..C-1]
            y_idx = torch.zeros_like(y)
            for lab, idx in label_to_idx.items():
                y_idx[y == lab] = idx

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(x, batch)
                loss = F.cross_entropy(logits, y_idx, weight=w)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            pred = logits.argmax(1)
            tot_correct += (pred == y_idx).sum().item()
            tot_n += y_idx.numel()
            tot_loss += loss.item()

        sch.step()
        train_acc = 100.0 * tot_correct / max(tot_n, 1)

        # ---- val ----
        model.eval()
        val_correct, val_n = 0, 0
        with torch.no_grad():
            for x, y, batch in tqdm(val_loader, desc=f"Val {ep}/{CFG['epochs']}", leave=False):
                x = x.to(CFG["device"], non_blocking=True)
                y = y.to(CFG["device"], non_blocking=True)
                batch = batch.to(CFG["device"], non_blocking=True)

                y_idx = torch.zeros_like(y)
                for lab, idx in label_to_idx.items():
                    y_idx[y == lab] = idx

                logits = model(x, batch)
                pred = logits.argmax(1)

                val_correct += (pred == y_idx).sum().item()
                val_n += y_idx.numel()

        val_acc = 100.0 * val_correct / max(val_n, 1)
        print(f"Epoch {ep:03d} | loss {tot_loss/len(train_loader):.4f} | train {train_acc:.2f}% | val {val_acc:.2f}%")

        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                "model": model.state_dict(),
                "in_ch": in_ch,
                "k": CFG["k"],
                "label_to_idx": label_to_idx,
                "idx_to_label": idx_to_label,
            }, CFG["save_path"])
            print("✅ Saved best:", CFG["save_path"])

    print("✅ Done. Best val:", best_val)

if __name__ == "__main__":
    main()