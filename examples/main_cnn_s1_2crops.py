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
from scipy.ndimage import gaussian_filter, sobel, label  # type: ignore
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

# -----------------------------
# Datasets: 1a + 1b
# -----------------------------
CROP_IDS_1A = [
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

CROP_IDS_1B = [
    "crop235",
    "crop240",
    "crop241",
    "crop242",
    "crop245",
    "crop249",
    "crop255",
    "crop258",
    "crop291",
]

RAW_1A_S1 = r"../data/jrc_cos7-1a/jrc_cos7-1a.zarr/recon-1/em/fibsem-uint8/s1"
GT_1A_ROOT = r"../data/jrc_cos7-1a/jrc_cos7-1a.zarr/recon-1/labels/groundtruth"

RAW_1B_S1 = r"../data/jrc_cos7-1b/jrc_cos7-1b.zarr/recon-1/em/fibsem-uint8/s1"
GT_1B_ROOT = r"../data/jrc_cos7-1b/jrc_cos7-1b.zarr/recon-1/labels/groundtruth"

# Fixed validation set: crop234 from 1a
VAL_DATASET = "1a"
VAL_CROP_ID = "crop234"

SELECT_CLASSES = {
    "cyto": 35,
    "mito_mem": 3,
    "mito_lum": 4,
    "er_mem": 16,
    "er_lum": 17,
}

# Model-internal IDs (continuous 0..5)
CLASS_ID_MAP = {
    "cyto": 1,
    "mito_mem": 2,
    "mito_lum": 3,
    "er_mem": 4,
    "er_lum": 5,
}

CLASS_NAMES = ["bg", "cyto", "mito_mem", "mito_lum", "er_mem", "er_lum"]
NUM_CLASSES = 6  # 0..5

# Visualization-only mapping: model id -> original CellMap label id
# This changes ONLY the legend text (and optional remap if you want).
VIS_LABEL_MAP = {
    0: 0,   # bg
    1: 35,  # cyto
    2: 3,   # mito_mem
    3: 4,   # mito_lum
    4: 16,  # er_mem
    5: 17,  # er_lum
}

# === CLASS WEIGHTS to address Dice=0 issues ===
# Weights: [bg, cyto, mito_mem, mito_lum, er_mem, er_lum]
CLASS_WEIGHTS_LIST = [3.0, 1.0, 10.0, 10.0, 5.0, 5.0]
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
INFER_STRIDE = (4, 32, 32)  # overlap = patch - stride

# Output
N_VAL_SLICES = 10
from datetime import datetime

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"../Result/unet3d_runs/run_{RUN_ID}"


# -----------------------------
# Feature extraction for CNN
# Recommended for 3D U-Net: 2 channels = [raw, gauss(s=2)]
# (Optional 3rd channel = gradmag2 if you want)
# -----------------------------
USE_GRADMAG2_AS_3RD_CH = False


def make_cnn_input(raw_uint8_zyx: np.ndarray) -> np.ndarray:
    """
    Input: raw uint8 (Z,Y,X)
    Output: float32 (C,Z,Y,X), C=2 or 3
    """
    img = raw_uint8_zyx.astype(np.float32) / 255.0
    g2 = gaussian_filter(img, sigma=2.0)

    if not USE_GRADMAG2_AS_3RD_CH:
        x = np.stack([img, g2], axis=0).astype(np.float32)
        return x

    # Optional: Grad magnitude on g2 (XY only)
    gx = sobel(g2, axis=2)
    gy = sobel(g2, axis=1)
    gradmag2 = np.sqrt(gx * gx + gy * gy)
    x = np.stack([img, g2, gradmag2], axis=0).astype(np.float32)
    return x


def downsample_mask_max(arr_s0: np.ndarray, factor: int = 2) -> np.ndarray:
    """
    Max-downsample for binary/label masks to preserve thin structures (membranes).
    arr_s0: (Z,Y,X) uint8/bool
    returns: (Z//factor, Y//factor, X//factor) uint8
    """
    if factor == 1:
        return arr_s0

    Z, Y, X = arr_s0.shape
    Z2, Y2, X2 = Z // factor, Y // factor, X // factor

    if Z2 <= 0 or Y2 <= 0 or X2 <= 0:
        raise ValueError(
            f"downsample_mask_max: volume too small for factor={factor}, shape={arr_s0.shape}"
        )

    arr = arr_s0[: Z2 * factor, : Y2 * factor, : X2 * factor]
    arr = arr.reshape(Z2, factor, Y2, factor, X2, factor)
    out = arr.max(axis=(1, 3, 5))
    return out.astype(np.uint8)


def load_one_crop(crop_id: str, raw_zarr, groundtruth_root: str, dataset_tag: str) -> dict:
    """
    Load s1 raw data and downsample s0 labels to match.
    Assumes s1 is 2x downsampled compared to s0.
    """
    scale_factor = 2  # s1 is s0 / 2
    crop_root = os.path.join(groundtruth_root, crop_id)

    # Reference class (for shape + transform) with fallback
    ref_s0 = os.path.join(crop_root, REF_CLASS, "s0")
    ref_zattr = os.path.join(crop_root, REF_CLASS, ".zattrs")

    if (not os.path.exists(ref_s0)) or (not os.path.exists(ref_zattr)):
        found = False
        for alt in SELECT_CLASSES.keys():
            alt_s0 = os.path.join(crop_root, alt, "s0")
            alt_zattr = os.path.join(crop_root, alt, ".zattrs")
            if os.path.exists(alt_s0) and os.path.exists(alt_zattr):
                ref_s0, ref_zattr = alt_s0, alt_zattr
                found = True
                print(f"[{dataset_tag}/{crop_id}] REF_CLASS '{REF_CLASS}' missing, fallback to '{alt}'")
                break

        if not found:
            raise FileNotFoundError(
                f"[{dataset_tag}/{crop_id}] Cannot find reference s0/.zattrs. "
                f"Tried REF_CLASS='{REF_CLASS}' and fallback classes={list(SELECT_CLASSES.keys())} "
                f"under {crop_root}"
            )

    # 1) Read metadata from s0 (Ground Truth definition)
    ref_arr = zarr.open(ref_s0, mode="r")
    Dz_s0, Dy_s0, Dx_s0 = ref_arr.shape

    with open(ref_zattr, "r") as f:
        attrs = json.load(f)

    ms = attrs["multiscales"][0]["datasets"][0]
    scale = ms["coordinateTransformations"][0]["scale"]
    trans = ms["coordinateTransformations"][1]["translation"]
    scale_z, scale_y, scale_x = scale
    tz, ty, tx = trans

    # 2) Calculate s0 indices
    vz0_s0 = int(tz / scale_z)
    vy0_s0 = int(ty / scale_y)
    vx0_s0 = int(tx / scale_x)

    # 3) Convert to s1 indices
    vz0 = vz0_s0 // scale_factor
    vy0 = vy0_s0 // scale_factor
    vx0 = vx0_s0 // scale_factor

    # Calculate s1 shape
    Dz = Dz_s0 // scale_factor
    Dy = Dy_s0 // scale_factor
    Dx = Dx_s0 // scale_factor

    vz1, vy1, vx1 = vz0 + Dz, vy0 + Dy, vx0 + Dx

    # 4) Load raw (from s1 path)
    raw_crop = raw_zarr[vz0:vz1, vy0:vy1, vx0:vx1]
    raw_crop = gaussian_filter(raw_crop.astype(np.float32), sigma=0.5).astype(np.uint8)

    # 5) Build multi-class label (load s0, then downsample to s1)
    label_multi = np.zeros((Dz, Dy, Dx), dtype=np.uint8)

    for cname in SELECT_CLASSES.keys():
        path = os.path.join(crop_root, cname, "s0")
        try:
            arr_s0 = zarr.open(path, mode="r")[:]
            arr_s1 = downsample_mask_max(arr_s0, factor=scale_factor)
            arr_s1 = arr_s1[:Dz, :Dy, :Dx]

            cid = CLASS_ID_MAP[cname]  # 1..5
            label_multi[arr_s1 > 0] = cid

        except Exception as e:
            print(f"Warning: failed load class {cname} in {dataset_tag}/{crop_id}: {e}")

    # Safety check
    if raw_crop.shape != label_multi.shape:
        print(f"[{dataset_tag}/{crop_id}] Shape Mismatch! Raw: {raw_crop.shape}, Label: {label_multi.shape}")
        mz = min(raw_crop.shape[0], label_multi.shape[0])
        my = min(raw_crop.shape[1], label_multi.shape[1])
        mx = min(raw_crop.shape[2], label_multi.shape[2])
        raw_crop = raw_crop[:mz, :my, :mx]
        label_multi = label_multi[:mz, :my, :mx]

    return {
        "raw": raw_crop,
        "label": label_multi,
        "shape": raw_crop.shape,
        "id": f"{dataset_tag}_{crop_id}",
    }


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
            assert Dz >= self.pz and Dy >= self.py and Dx >= self.px, f"Crop too small: {c['id']}"
            self.ranges.append((Dz - self.pz, Dy - self.py, Dx - self.px))

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        # Pick a crop
        ci = np.random.randint(0, len(self.crops))
        crop = self.crops[ci]
        Dz_off, Dy_off, Dx_off = self.ranges[ci]

        z0 = np.random.randint(0, Dz_off + 1)
        y0 = np.random.randint(0, Dy_off + 1)
        x0 = np.random.randint(0, Dx_off + 1)

        raw_patch = crop["raw"][z0 : z0 + self.pz, y0 : y0 + self.py, x0 : x0 + self.px]
        y_patch = crop["label"][z0 : z0 + self.pz, y0 : y0 + self.py, x0 : x0 + self.px]

        # Ensure patch Z size is correct
        current_pz = raw_patch.shape[0]
        if current_pz != self.pz:
            print(f"Warning: Patch size mismatch (Z={current_pz} != {self.pz}). Resampling...")
            return self.__getitem__(np.random.randint(0, self.n_patches))

        # Ensure Y and X are also correct
        assert raw_patch.shape[1] == self.py and raw_patch.shape[2] == self.px, "Y or X dimension mismatch."

        x_patch = make_cnn_input(raw_patch)  # (C,Z,Y,X)

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

    dice_loss_per_class = 1.0 - dice

    if weights is not None:
        weights = weights.to(dice_loss_per_class.device)
        weighted_loss = weights * dice_loss_per_class
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


def remove_small_components(seg, min_voxels=2000):
    """
    seg: (Z,Y,X) uint8 segmentation
    Remove small isolated components for each foreground class.
    """
    out = seg.copy()
    for c in range(1, NUM_CLASSES):  # skip background
        mask = seg == c
        lab, n = label(mask)
        for i in range(1, n + 1):
            if (lab == i).sum() < min_voxels:
                out[lab == i] = 0
    return out


# -----------------------------
# High-Quality Sliding-window inference (Gaussian + Mirror Padding)
# -----------------------------
def _get_gaussian(patch_size, sigma_scale=1.0 / 8):
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode="constant", cval=0)

    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)

    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0]
    )
    return gaussian_importance_map


@torch.no_grad()
def sliding_window_predict(
    model, raw_zyx: np.ndarray, patch_zyx=(32, 128, 128), stride_zyx=(16, 64, 64)
):
    """
    Inference: Gaussian blending + mirror padding + simple flip TTA (4x).
    """
    model.eval()

    pz, py, px = patch_zyx
    pad_z, pad_y, pad_x = pz // 2, py // 2, px // 2

    raw_padded = np.pad(
        raw_zyx, ((pad_z, pad_z), (pad_y, pad_y), (pad_x, pad_x)), mode="reflect"
    )

    Dz_pad, Dy_pad, Dx_pad = raw_padded.shape
    sz, sy, sx = stride_zyx

    scores = np.zeros((NUM_CLASSES, Dz_pad, Dy_pad, Dx_pad), dtype=np.float32)
    gaussian_weights = np.zeros((Dz_pad, Dy_pad, Dx_pad), dtype=np.float32)

    g_window = _get_gaussian(patch_zyx)
    g_window_torch = torch.from_numpy(g_window).to(DEVICE)

    # Input tensor: (B, C, Z, Y, X) -> flip dims: 2(Z), 3(Y), 4(X)
    tta_dims = [[], [2], [3], [4]]

    z_starts = list(range(0, max(Dz_pad - pz, 0) + 1, sz))
    y_starts = list(range(0, max(Dy_pad - py, 0) + 1, sy))
    x_starts = list(range(0, max(Dx_pad - px, 0) + 1, sx))

    if z_starts[-1] != Dz_pad - pz:
        z_starts.append(Dz_pad - pz)
    if y_starts[-1] != Dy_pad - py:
        y_starts.append(Dy_pad - py)
    if x_starts[-1] != Dx_pad - px:
        x_starts.append(Dx_pad - px)

    for z0 in tqdm(z_starts, desc="Infer (TTA+Gaussian)"):
        for y0 in y_starts:
            for x0 in x_starts:
                patch_raw = raw_padded[z0 : z0 + pz, y0 : y0 + py, x0 : x0 + px]
                if patch_raw.shape != (pz, py, px):
                    continue

                x_in_np = make_cnn_input(patch_raw)  # (C, Z, Y, X)
                x_in = torch.from_numpy(x_in_np).unsqueeze(0).to(DEVICE)

                patch_prob_sum = torch.zeros((NUM_CLASSES, pz, py, px), device=DEVICE)

                for dims in tta_dims:
                    if len(dims) == 0:
                        logits = model(x_in)
                    else:
                        x_flipped = torch.flip(x_in, dims=dims)
                        logits_flipped = model(x_flipped)
                        logits = torch.flip(logits_flipped, dims=dims)

                    prob = torch.softmax(logits, dim=1).squeeze(0)  # (K, Z, Y, X)
                    patch_prob_sum += prob

                avg_prob = patch_prob_sum / len(tta_dims)
                avg_prob *= g_window_torch

                scores[:, z0 : z0 + pz, y0 : y0 + py, x0 : x0 + px] += avg_prob.cpu().numpy()
                gaussian_weights[z0 : z0 + pz, y0 : y0 + py, x0 : x0 + px] += g_window

    gaussian_weights = np.maximum(gaussian_weights, 1e-6)
    scores /= gaussian_weights[None, ...]
    scores = scores[:, pad_z:-pad_z, pad_y:-pad_y, pad_x:-pad_x]

    return np.argmax(scores, axis=0).astype(np.uint8)


# -----------------------------
# Train one fold
# -----------------------------
def train_one_fold(fold_id, train_crops, val_crops):
    run_dir = os.path.join(OUT_DIR)
    os.makedirs(run_dir, exist_ok=True)

    in_ch = 3 if USE_GRADMAG2_AS_3RD_CH else 2
    model = UNet3D(in_ch=in_ch, n_classes=NUM_CLASSES, base=32).to(DEVICE)

    class_weights = torch.tensor(CLASS_WEIGHTS_LIST, dtype=torch.float32).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    ce = nn.CrossEntropyLoss(weight=class_weights)

    train_ds = RandomPatchDataset(train_crops, n_patches=PATCHES_PER_EPOCH, patch_zyx=PATCH_ZYX)
    val_ds = RandomPatchDataset(val_crops, n_patches=VAL_PATCHES, patch_zyx=PATCH_ZYX)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    best_val = 1e9
    best_path = os.path.join(run_dir, "best.pt")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        tr_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"Train epoch {epoch}", leave=False):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = 0.5 * ce(logits, y) + 0.5 * soft_dice_loss(logits, y, num_classes=NUM_CLASSES)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()

        tr_loss /= max(len(train_loader), 1)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Val epoch {epoch}", leave=False):
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                logits = model(x)
                loss = 0.5 * ce(logits, y) + 0.5 * soft_dice_loss(logits, y, num_classes=NUM_CLASSES)
                va_loss += loss.item()

        va_loss /= max(len(val_loader), 1)
        dt = time.time() - t0

        print(f"Epoch {epoch:03d} | train={tr_loss:.4f} val={va_loss:.4f} | {dt/60:.2f} min")

        if va_loss < best_val:
            best_val = va_loss
            torch.save({"model": model.state_dict(), "in_ch": in_ch}, best_path)

    print(f"Best val loss: {best_val:.4f}")

    # Full-volume inference on validation crop
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
    pred = remove_small_components(pred, min_voxels=2000)

    gt = val_crop["label"]

    Z = min(pred.shape[0], gt.shape[0])
    Y = min(pred.shape[1], gt.shape[1])
    X = min(pred.shape[2], gt.shape[2])
    pred = pred[:Z, :Y, :X]
    gt = gt[:Z, :Y, :X]

    dices = dice_per_class(pred, gt, num_classes=NUM_CLASSES)
    print("Full-volume Dice per class:")
    for ci, d in enumerate(dices):
        print(f"  {ci}:{CLASS_NAMES[ci]}  Dice={d:.4f}")

    # Save slice visualizations
    vis_dir = os.path.join(run_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    Dz = val_crop["raw"].shape[0]
    zs = np.linspace(0, Dz - 1, N_VAL_SLICES, dtype=int)

    # Colormap and legend patches
    # NOTE: We keep actual pixel values 0..5 for rendering,
    # only the legend text shows original CellMap IDs via VIS_LABEL_MAP.
    cmap = plt.get_cmap("tab10")

    # Legend order to match the screenshot:
    # bg, mito_mem, mito_lum, er_mem, er_lum, cyto
    LEGEND_ORDER = [0, 2, 3, 4, 5, 1]  # model IDs in desired legend order

    legend_patches = []
    for cid in LEGEND_ORDER:
        name = CLASS_NAMES[cid]                 # class name by model id
        show_id = VIS_LABEL_MAP.get(cid, cid)   # original CellMap id for display
        legend_patches.append(
            mpatches.Patch(color=cmap(cid), label=f"{name} ({show_id})")
        )

    for z in zs:
        fig = plt.figure(figsize=(20, 6))

        # Raw
        plt.subplot(1, 3, 1)
        plt.title("Raw")
        plt.imshow(val_crop["raw"][z], cmap="gray")
        plt.axis("off")

        # GT
        plt.subplot(1, 3, 2)
        plt.title("GT")
        plt.imshow(val_crop["label"][z], cmap=cmap, vmin=0, vmax=9)
        plt.axis("off")
        plt.legend(
            handles=legend_patches,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
            title="Classes (ID)",
        )

        # Pred
        plt.subplot(1, 3, 3)
        plt.title("Pred")
        plt.imshow(pred[z], cmap=cmap, vmin=0, vmax=9)
        plt.axis("off")
        plt.legend(
            handles=legend_patches,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
            title="Classes (ID)",
        )

        plt.tight_layout()
        outp = os.path.join(vis_dir, f"val_{val_crop['id']}_z{z:04d}.png")
        plt.savefig(outp, dpi=200)
        plt.close(fig)

    return best_val, dices


def main():
    print("DEVICE:", DEVICE)

    # Open both datasets raw (s1)
    raw_1a = zarr.open(RAW_1A_S1, mode="r")
    raw_1b = zarr.open(RAW_1B_S1, mode="r")
    print("Raw 1a shape:", raw_1a.shape)
    print("Raw 1b shape:", raw_1b.shape)

    # Load crops from 1a
    crops_1a = []
    print("\n===== Loading 1a crops =====")
    for cid in CROP_IDS_1A:
        print(f"Loading 1a {cid}...")
        c = load_one_crop(cid, raw_1a, GT_1A_ROOT, dataset_tag="1a")
        print("  shape:", c["shape"], "labels:", np.unique(c["label"]))
        crops_1a.append(c)

    # Load crops from 1b
    crops_1b = []
    print("\n===== Loading 1b crops =====")
    for cid in CROP_IDS_1B:
        print(f"Loading 1b {cid}...")
        c = load_one_crop(cid, raw_1b, GT_1B_ROOT, dataset_tag="1b")
        print("  shape:", c["shape"], "labels:", np.unique(c["label"]))
        crops_1b.append(c)

    # Fixed validation: 1a crop234
    val_key = f"1a_{VAL_CROP_ID}"
    val_crops = [c for c in crops_1a if c["id"] == val_key]
    assert len(val_crops) == 1, f"Validation crop not found: {val_key}"

    # Training crops: (1a except crop234) + (all 1b)
    train_crops = [c for c in crops_1a if c["id"] != val_key] + crops_1b

    print("\n" + "=" * 60)
    print(
        f"SINGLE FOLD (mixed 1a+1b) | "
        f"train={[c['id'] for c in train_crops]} | "
        f"val={[c['id'] for c in val_crops]}"
    )

    fold_id = 0
    loss, dices = train_one_fold(fold_id, train_crops, val_crops)

    print("\n===== Done =====")
    print("Best val loss:", loss)


if __name__ == "__main__":
    main()
