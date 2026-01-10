# =============================================
# 3D U-Net for CellMap multi-class segmentation (crop-based KFold)
# Input channels: Raw + Gaussian(sigma=2)  (recommended for 3D U-Net)
# Labels: background(0) + 5 classes (1..5)
# Data loading aligned with your .zattrs method
# =============================================

import os
import json
import time
import random
import numpy as np
import zarr  # type: ignore
from tqdm import tqdm  # type: ignore
from scipy.ndimage import gaussian_filter, sobel  # type: ignore
from sklearn.model_selection import KFold  # type: ignore
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.patches as mpatches


# -----------------------------
# Config
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Crops
CROP_IDS = [
    "crop234",
    "crop236",
    "crop237",
    "crop239",
    "crop248",
    "crop252",
    "crop254",
    "crop256",
    "crop292",
]

RAW_S0 = r"../data/jrc_cos7-1a/jrc_cos7-1a.zarr/recon-1/em/fibsem-uint8/s0"
GROUNDTRUTH_ROOT = r"../data/jrc_cos7-1a/jrc_cos7-1a.zarr/recon-1/labels/groundtruth"

SELECT_CLASSES = {
    "cyto": 35,
    "mito_mem": 3,
    "mito_lum": 4,
    "er_mem": 16,
    "er_lum": 17,
}

CLASS_ID_MAP = {
    "cyto": 1,
    "mito_mem": 2,
    "mito_lum": 3,
    "er_mem": 4,
    "er_lum": 5,
}

CLASS_NAMES = ["bg", "cyto", "mito_mem", "mito_lum", "er_mem", "er_lum"]
NUM_CLASSES = 6  # 0..5

# === NEW: CLASS WEIGHTS to address Dice=0 issues ===
# Weights: [bg, cyto, mito_mem, mito_lum, er_mem, er_lum]
CLASS_WEIGHTS_LIST = [1.0, 1.0, 10.0, 10.0, 5.0, 5.0]
# ===================================================

REF_CLASS = "nucpl"

# Training
EPOCHS = 300
BATCH_SIZE = 2
LR = 2e-4
WEIGHT_DECAY = 1e-5

# Patch sampling (adjust to your GPU memory)
PATCH_ZYX = (32, 128, 128)  # (Z,Y,X)
PATCHES_PER_EPOCH = 300  # per fold
VAL_PATCHES = 80

# Sliding-window inference for full volume evaluation
INFER_STRIDE = (16, 64, 64)  # overlap = patch - stride

# Output
N_VAL_SLICES = 10
from datetime import datetime

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"../Result/unet3d_runs/run_{RUN_ID}"
# os.makedirs(OUT_DIR, exist_ok=True)
# OUT_DIR = "../Result/unet3d_runs"
# os.makedirs(OUT_DIR, exist_ok=True)


# -----------------------------
# Feature extraction for CNN
# Recommended for 3D U-Net: 2 channels = [raw, gauss(s=2)]
# (Optional 3rd channel = gradmag2 if you want)
# -----------------------------
USE_GRADMAG2_AS_3RD_CH = False


def make_cnn_input(raw_uint8_zyx: np.ndarray) -> np.ndarray:
    """
    Input: raw uint8 (Z,Y,X)
    Output: float32 (C,Z,Y,X)
      C=2 or 3
    """
    img = raw_uint8_zyx.astype(np.float32) / 255.0  # raw
    g2 = gaussian_filter(img, sigma=2.0)

    if not USE_GRADMAG2_AS_3RD_CH:
        x = np.stack([img, g2], axis=0).astype(np.float32)
        return x

    # Optional: GradMag on g2 (XY only)
    gx = sobel(g2, axis=2)
    gy = sobel(g2, axis=1)
    gradmag2 = np.sqrt(gx * gx + gy * gy)
    x = np.stack([img, g2, gradmag2], axis=0).astype(np.float32)
    return x


# -----------------------------
# Data loading (aligned to your method)
# -----------------------------
def load_one_crop(crop_id: str, raw_zarr) -> dict:
    """
    Returns dict:
      raw: uint8 (Z,Y,X)
      label: uint8 (Z,Y,X) 0..5
      id: str
      shape: (Z,Y,X)
    """
    crop_root = os.path.join(GROUNDTRUTH_ROOT, crop_id)
    ref_s0 = os.path.join(crop_root, REF_CLASS, "s0")
    ref_zattr = os.path.join(crop_root, REF_CLASS, ".zattrs")

    ref_arr = zarr.open(ref_s0, mode="r")
    Dz, Dy, Dx = ref_arr.shape

    with open(ref_zattr, "r") as f:
        attrs = json.load(f)
    ms = attrs["multiscales"][0]["datasets"][0]
    scale = ms["coordinateTransformations"][0]["scale"]
    trans = ms["coordinateTransformations"][1]["translation"]
    scale_z, scale_y, scale_x = scale
    tz, ty, tx = trans

    vz0 = int(tz / scale_z)
    vy0 = int(ty / scale_y)
    vx0 = int(tx / scale_x)
    vz1, vy1, vx1 = vz0 + Dz, vy0 + Dy, vx0 + Dx

    raw_crop = raw_zarr[vz0:vz1, vy0:vy1, vx0:vx1]  # uint8 (Z,Y,X)

    # Build multi-class label
    label_multi = np.zeros((Dz, Dy, Dx), dtype=np.uint8)
    for cname in SELECT_CLASSES.keys():
        path = os.path.join(crop_root, cname, "s0")
        try:
            arr = zarr.open(path, mode="r")[:]  # binary mask
            cid = CLASS_ID_MAP[cname]
            label_multi[arr > 0] = cid
        except Exception as e:
            print(f"Warning: failed load class {cname} in {crop_id}: {e}")

    return {"raw": raw_crop, "label": label_multi, "shape": (Dz, Dy, Dx), "id": crop_id}


# -----------------------------
# Patch sampler dataset
# -----------------------------
class RandomPatchDataset(Dataset):
    def __init__(self, crops: list[dict], n_patches: int, patch_zyx=(32, 128, 128)):
        self.crops = crops
        self.n_patches = n_patches
        self.pz, self.py, self.px = patch_zyx

        # Precompute per-crop valid ranges
        self.ranges = []
        for c in self.crops:
            Dz, Dy, Dx = c["shape"]
            assert (
                Dz >= self.pz and Dy >= self.py and Dx >= self.px
            ), f"Crop too small: {c['id']}"
            self.ranges.append((Dz - self.pz, Dy - self.py, Dx - self.px))

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        # pick a crop
        ci = np.random.randint(0, len(self.crops))
        crop = self.crops[ci]
        Dz_off, Dy_off, Dx_off = self.ranges[ci]

        z0 = np.random.randint(0, Dz_off + 1)
        y0 = np.random.randint(0, Dy_off + 1)
        x0 = np.random.randint(0, Dx_off + 1)

        raw_patch = crop["raw"][z0 : z0 + self.pz, y0 : y0 + self.py, x0 : x0 + self.px]
        y_patch = crop["label"][z0 : z0 + self.pz, y0 : y0 + self.py, x0 : x0 + self.px]

        # -------------------
        # ✅ modifyd here: ensure patch size is correct
        # -------------------
        current_pz = raw_patch.shape[0]
        if current_pz != self.pz:
             print(f"Warning: Patch size mismatch (Z={current_pz} != {self.pz}). Resampling...")
             return self.__getitem__(np.random.randint(0, self.n_patches)) 
             
        # Optional: ensure Y and X are also correct
        assert raw_patch.shape[1] == self.py and raw_patch.shape[2] == self.px, "Y or X dimension mismatch after slicing."
        # -------------------

        x_patch = make_cnn_input(raw_patch)  # (C,Z,Y,X)

        # torch
        x = torch.from_numpy(x_patch)  # float32
        y = torch.from_numpy(y_patch.astype(np.int64))  # long (Z,Y,X)
        return x, y


# -----------------------------
# 3D U-Net (small & clean)
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet3D(nn.Module):
    def __init__(self, in_ch, n_classes, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool3d(2)

        self.bott = DoubleConv(base * 4, base * 8)

        self.up3 = nn.ConvTranspose3d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.out = nn.Conv3d(base, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bott(self.pool3(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)


# -----------------------------
# Loss: CE + Soft Dice (multi-class)
# -----------------------------
def soft_dice_loss(logits, targets, num_classes=6, weights=None, eps=1e-6):
    """
    logits: (B, C, Z, Y, X)
    targets: (B, Z, Y, X) int
    """
    probs = torch.softmax(logits, dim=1)
    onehot = F.one_hot(targets, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

    dims = (0, 2, 3, 4)
    inter = torch.sum(probs * onehot, dims)
    denom = torch.sum(probs + onehot, dims)
    dice = (2 * inter + eps) / (denom + eps)

    # NEW: Loss per class is 1 - Dice
    dice_loss_per_class = 1.0 - dice

    if weights is not None:
        weights = weights.to(dice_loss_per_class.device)
        # Apply weights to the loss-per-class (1 - Dice)
        weighted_loss = weights * dice_loss_per_class
        # Normalized weighted mean of the loss
        loss = weighted_loss.sum() / weights.sum()
    else:
        loss = dice_loss_per_class.mean()

    return loss


# -----------------------------
# Metrics (Dice per class)
# -----------------------------
@torch.no_grad()
def dice_per_class(pred, gt, num_classes=6, eps=1e-6):
    """
    pred, gt: (Z,Y,X) int (cpu numpy or torch)
    returns: list length num_classes
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()

    out = []
    for c in range(num_classes):
        p = pred == c
        g = gt == c
        inter = (p & g).sum()
        denom = p.sum() + g.sum()
        out.append((2 * inter + eps) / (denom + eps))
    return out


# -----------------------------
# High-Quality Sliding-window inference (Gaussian + Mirror Padding)
# -----------------------------
def _get_gaussian(patch_size, sigma_scale=1.0 / 8):
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    
    # Normalize
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)

    # Gaussian cannot be 0, otherwise division by zero errors
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0]
    )
    return gaussian_importance_map

@torch.no_grad()
def sliding_window_predict(
    model, raw_zyx: np.ndarray, patch_zyx=(32, 128, 128), stride_zyx=(16, 64, 64)
):
    """
    SOTA Inference with Gaussian Blending and Mirror Padding.
    Eliminates edge artifacts.
    """
    model.eval()
    
    # 1. Pad the input volume to handle edges
    # We pad by half the patch size to ensure edge pixels can be in the center of a patch
    pz, py, px = patch_zyx
    pad_z, pad_y, pad_x = pz // 2, py // 2, px // 2
    
    # Use 'reflect' to avoid sharp zero-boundaries that confuse InstanceNorm
    raw_padded = np.pad(
        raw_zyx, 
        ((pad_z, pad_z), (pad_y, pad_y), (pad_x, pad_x)), 
        mode='reflect'
    )
    
    Dz_pad, Dy_pad, Dx_pad = raw_padded.shape
    sz, sy, sx = stride_zyx

    # 2. Prepare result arrays
    scores = np.zeros((NUM_CLASSES, Dz_pad, Dy_pad, Dx_pad), dtype=np.float32)
    # Instead of counting '1', we accumulate the gaussian weights
    gaussian_weights = np.zeros((Dz_pad, Dy_pad, Dx_pad), dtype=np.float32)

    # 3. Get Gaussian window
    # This window gives high weight to center, low weight to edges
    g_window = _get_gaussian(patch_zyx)
    g_window_torch = torch.from_numpy(g_window).to(DEVICE) # (Z,Y,X)

    # 4. Sliding locations
    z_starts = list(range(0, max(Dz_pad - pz, 0) + 1, sz))
    y_starts = list(range(0, max(Dy_pad - py, 0) + 1, sy))
    x_starts = list(range(0, max(Dx_pad - px, 0) + 1, sx))
    
    # Ensure we cover the last bit
    if z_starts[-1] != Dz_pad - pz: z_starts.append(Dz_pad - pz)
    if y_starts[-1] != Dy_pad - py: y_starts.append(Dy_pad - py)
    if x_starts[-1] != Dx_pad - px: x_starts.append(Dx_pad - px)

    # 5. Inference Loop
    for z0 in tqdm(z_starts, desc="Infer (Gaussian)"):
        for y0 in y_starts:
            for x0 in x_starts:
                # Extract patch
                patch_raw = raw_padded[z0 : z0 + pz, y0 : y0 + py, x0 : x0 + px]
                
                # Check shape (sanity check)
                if patch_raw.shape != (pz, py, px):
                    continue

                x_patch = make_cnn_input(patch_raw)  # (C,Z,Y,X)
                x = torch.from_numpy(x_patch).unsqueeze(0).to(DEVICE)  # (1,C,Z,Y,X)

                logits = model(x)
                prob = torch.softmax(logits, dim=1).squeeze(0) # (K,Z,Y,X)

                # --- The Magic: Multiply by Gaussian Window ---
                prob *= g_window_torch
                
                # Accumulate
                prob_np = prob.cpu().numpy()
                scores[:, z0 : z0 + pz, y0 : y0 + py, x0 : x0 + px] += prob_np
                gaussian_weights[z0 : z0 + pz, y0 : y0 + py, x0 : x0 + px] += g_window

    # 6. Normalize and Crop back
    # Avoid division by zero
    gaussian_weights = np.maximum(gaussian_weights, 1e-6)
    scores /= gaussian_weights[None, ...]
    
    # Crop back to original size
    scores = scores[:, pad_z:-pad_z, pad_y:-pad_y, pad_x:-pad_x]
    
    pred = np.argmax(scores, axis=0).astype(np.uint8)
    return pred


# -----------------------------
# Train one fold
# -----------------------------
def train_one_fold(fold_id, train_crops, val_crops):
    # run_dir = os.path.join(OUT_DIR, f"fold_{fold_id}")
    run_dir = os.path.join(OUT_DIR)
    os.makedirs(run_dir, exist_ok=True)

    in_ch = 3 if USE_GRADMAG2_AS_3RD_CH else 2
    model = UNet3D(in_ch=in_ch, n_classes=NUM_CLASSES, base=32).to(DEVICE)

    class_weights = torch.tensor(CLASS_WEIGHTS_LIST, dtype=torch.float32).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    ce = nn.CrossEntropyLoss(weight=class_weights)

    train_ds = RandomPatchDataset(
        train_crops, n_patches=PATCHES_PER_EPOCH, patch_zyx=PATCH_ZYX
    )
    val_ds = RandomPatchDataset(
        val_crops, n_patches=VAL_PATCHES, patch_zyx=PATCH_ZYX
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True
    )

    SAVE_EVERY = 50
    best_val = 1e9
    best_path = os.path.join(run_dir, "best.pt")

    # ===========================
    # training loop
    # ===========================
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        tr_loss = 0.0

        for x, y in tqdm(
            train_loader,
            desc=f"Train epoch {epoch}",
            leave=False,
        ):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = 0.5 * ce(logits, y) + 0.5 * soft_dice_loss(
                logits, y, num_classes=NUM_CLASSES
            )
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()

        tr_loss /= max(len(train_loader), 1)

        # ---------- validation ----------
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for x, y in tqdm(
                val_loader,
                desc=f"Val epoch {epoch}",
                leave=False,
            ):
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                logits = model(x)
                loss = 0.5 * ce(logits, y) + 0.5 * soft_dice_loss(
                    logits, y, num_classes=NUM_CLASSES
                )
                va_loss += loss.item()

        va_loss /= max(len(val_loader), 1)
        dt = time.time() - t0

        print(
            f"Epoch {epoch:03d} | "
            f"train={tr_loss:.4f} val={va_loss:.4f} | {dt/60:.2f} min"
        )

        # ---------- best model ----------
        if va_loss < best_val:
            best_val = va_loss
            torch.save(
                {"model": model.state_dict(), "in_ch": in_ch},
                best_path,
            )

        # ---------- periodic checkpoint ----------
        if epoch % SAVE_EVERY == 0:
            ckpt_path = os.path.join(run_dir, f"epoch_{epoch:03d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "in_ch": in_ch,
                },
                ckpt_path,
            )

    print(f"Best val loss: {best_val:.4f}")

    # ===========================
    # full-volume inference on validation crop
    # ===========================
    ckpt = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    val_crop = val_crops[0]

    pred = sliding_window_predict(
        model,
        val_crop["raw"],
        patch_zyx=PATCH_ZYX,
        stride_zyx=INFER_STRIDE,
    )

    gt = val_crop["label"]

    Z = min(pred.shape[0], gt.shape[0])
    Y = min(pred.shape[1], gt.shape[1])
    X = min(pred.shape[2], gt.shape[2])
    pred = pred[:Z, :Y, :X]
    gt = gt[:Z, :Y, :X]

    dices = dice_per_class(pred, gt, num_classes=NUM_CLASSES)
    print(f"Full-volume Dice per class:")
    for ci, d in enumerate(dices):
        print(f"  {ci}:{CLASS_NAMES[ci]}  Dice={d:.4f}")

    # ===========================
    # save slice visualizations
    # ===========================
    vis_dir = os.path.join(run_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    Dz = val_crop["raw"].shape[0]
    zs = np.linspace(0, Dz - 1, N_VAL_SLICES, dtype=int)

    # 1. Setup Colormap and Legend Patches
    # Since your labels are already 0..5, we map them directly to tab10 colors
    cmap = plt.get_cmap("tab10")
    
    legend_patches = []
    for i, name in enumerate(CLASS_NAMES):
        # i is 0..5, matches the pixel value in GT/Pred
        legend_patches.append(mpatches.Patch(color=cmap(i), label=f"{name} ({i})"))

    for z in zs:
        # Increase figure width to make room for legend
        fig = plt.figure(figsize=(20, 6))

        # --- Raw ---
        plt.subplot(1, 3, 1)
        plt.title("Raw")
        plt.imshow(val_crop["raw"][z], cmap="gray")
        plt.axis("off")

        # --- GT ---
        plt.subplot(1, 3, 2)
        plt.title("GT")
        plt.imshow(val_crop["label"][z], cmap=cmap, vmin=0, vmax=9)
        plt.axis("off")
        # Add Legend
        plt.legend(
            handles=legend_patches,
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            borderaxespad=0.,
            title="Classes"
        )

        # --- Pred ---
        plt.subplot(1, 3, 3)
        plt.title("Pred")
        plt.imshow(pred[z], cmap=cmap, vmin=0, vmax=9)
        plt.axis("off")
        # Add Legend
        plt.legend(
            handles=legend_patches,
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            borderaxespad=0.,
            title="Classes"
        )

        plt.tight_layout()
        outp = os.path.join(
            vis_dir, f"val_{val_crop['id']}_z{z:04d}.png"
        )
        plt.savefig(outp, dpi=200)
        plt.close(fig)

    return best_val, dices


def main():
    print("DEVICE:", DEVICE)
    raw_zarr = zarr.open(RAW_S0, mode="r")
    print("Raw shape:", raw_zarr.shape)

    # -------- load crops --------
    all_crops = []
    print("\n===== Loading crops =====")
    for cid in CROP_IDS:
        print(f"Loading {cid}...")
        c = load_one_crop(cid, raw_zarr)
        print("  shape:", c["shape"], "labels:", np.unique(c["label"]))
        all_crops.append(c)

    # ===== single fold =====
    val_idx = 0  # select valid crop
    train_idx = [i for i in range(len(all_crops)) if i != val_idx]

    train_crops = [all_crops[i] for i in train_idx]
    val_crops   = [all_crops[val_idx]]

    print("\n" + "=" * 60)
    print(
        f"SINGLE FOLD | "
        f"train={[c['id'] for c in train_crops]} | "
        f"val={[c['id'] for c in val_crops]}"
    )

    fold_id = 0
    loss, dices = train_one_fold(fold_id, train_crops, val_crops)

    print("\n===== Done =====")
    print("Best val loss:", loss)


if __name__ == "__main__":
    main()
