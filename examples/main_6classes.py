# =============================================
#   CellMap nucleus-related segmentation (crop292 only)
#   Classes: 20,21,22,23,24,28
#   Author: Chenle Wang
#   TU Dresden - CMS Research Project
# =============================================

import os
import json
import numpy as np
import zarr 
import matplotlib.pyplot as plt 
from scipy.ndimage import gaussian_filter, sobel, laplace
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score, f1_score
from sklearn.model_selection import KFold 


# ---------------------------------------------
# Step 1. Setup paths
# ---------------------------------------------

RAW_S0 = r"../data/jrc_cos7-1a/jrc_cos7-1a.zarr/recon-1/em/fibsem-uint8/s0"
CROP292_ROOT = r"../data/jrc_cos7-1a/jrc_cos7-1a.zarr/recon-1/labels/groundtruth/crop292"

# 6 个核相关 atomic class
NUCLEUS_CLASSES = [
    "ne_mem",  # 20
    "ne_lum",  # 21
    "np_out",  # 22
    "np_in",  # 23
    "hchrom",  # 24
    "nucpl",  # 28
]

# multi-class 映射：背景=0
CLASS_ID_MAP = {
    "ne_mem": 1,
    "ne_lum": 2,
    "np_out": 3,
    "np_in": 4,
    "hchrom": 5,
    "nucpl": 6,
}

# 用 nucpl 做 reference（也可以自动选，这里简单固定）
REF_CLASS = "nucpl"
REF_S0 = os.path.join(CROP292_ROOT, REF_CLASS, "s0")
REF_ZATTR = os.path.join(CROP292_ROOT, REF_CLASS, ".zattrs")

raw_zarr = zarr.open(RAW_S0, mode="r")
ref_zarr = zarr.open(REF_S0, mode="r")

print("Raw shape:", raw_zarr.shape)
print("Reference label shape:", ref_zarr.shape)

# ---------------------------------------------
# Step 2. Read .zattrs (translation + scale)
# ---------------------------------------------

with open(REF_ZATTR, "r") as f:
    attrs = json.load(f)

ms = attrs["multiscales"][0]["datasets"][0]

scale = ms["coordinateTransformations"][0]["scale"]
trans = ms["coordinateTransformations"][1]["translation"]

scale_z, scale_y, scale_x = scale
tz, ty, tx = trans

print("Scale (nm):", scale)
print("Translation (nm):", trans)

# ---------------------------------------------
# Step 3. Convert nm → voxel index & crop raw
# ---------------------------------------------

vz0 = int(tz / scale_z)
vy0 = int(ty / scale_y)
vx0 = int(tx / scale_x)

Dz, Dy, Dx = ref_zarr.shape
vz1, vy1, vx1 = vz0 + Dz, vy0 + Dy, vx0 + Dx

print("Voxel coords:", (vz0, vz1), (vy0, vy1), (vx0, vx1))

raw_crop = raw_zarr[vz0:vz1, vy0:vy1, vx0:vx1]
print("Raw crop shape:", raw_crop.shape)

# ---------------------------------------------
# Step 4. Build multi-class label (0–6) + 可视化
# ---------------------------------------------

label_multi = np.zeros(ref_zarr.shape, dtype=np.uint8)

print("\n===== Loading 6-class nucleus labels =====")
for cname in NUCLEUS_CLASSES:
    path = os.path.join(CROP292_ROOT, cname, "s0")
    arr = zarr.open(path, mode="r")[:]
    cid = CLASS_ID_MAP[cname]
    print(f"{cname:7s} -> class {cid}, unique={np.unique(arr)}")
    label_multi[arr > 0] = cid

print("Unique final multi-class labels:", np.unique(label_multi))

# -------------------------------
# Step 4 Visualization (new)
# -------------------------------
# mid = label_multi.shape[0] // 2 + 1
# plt.figure(figsize=(18, 10))

# # 逐类单独可视化
# for i, cname in enumerate(NUCLEUS_CLASSES):
#     cid = CLASS_ID_MAP[cname]
#     mask = (label_multi[mid] == cid)

#     plt.subplot(2, 4, i + 1)
#     plt.imshow(mask, cmap="gray")
#     plt.title(f"{cname} (class {cid})")
#     plt.axis("off")

# # 多类别总览图（彩色）
# plt.subplot(2, 4, 7)
# plt.imshow(label_multi[mid], cmap="tab10")
# plt.title("Multi-class nucleus structures (6 classes)")
# plt.axis("off")

# # 叠加图（raw + label）
# plt.subplot(2, 4, 8)
# plt.imshow(raw_crop[mid], cmap='gray')
# plt.imshow(label_multi[mid], cmap='tab10', alpha=0.45)
# plt.title("Overlay: Raw + 6-class mask")
# plt.axis("off")

# plt.tight_layout()
# plt.show()

# ============================================================
# Step 5 (Modified): 3D Feature Extraction
# 三维特征提取：对每个 z-slice 提取前后层的空间信息
# ============================================================
import numpy as np


def extract_3d_features(volume, z):
    """
    volume: 3D EM volume (Z,Y,X)
    z: target slice index

    return: (Y,X,C) feature map | C=3 (raw, gaussian, sobel_mag)
    """

    # -------- 1) raw slice --------
    raw_ = volume[z]

    # -------- 2) 3D Gaussian (sigma=1) --------
    g1 = gaussian_filter(volume, sigma=1)[z]

    # -------- 3) 3D Sobel magnitude --------
    sob_z = sobel(volume, axis=0)[z]
    sob_y = sobel(volume, axis=1)[z]
    sob_x = sobel(volume, axis=2)[z]
    sob_mag = np.sqrt(sob_x**2 + sob_y**2 + sob_z**2)

    # -------- stack 3 feature channels --------
    feat = np.stack([raw_, g1, sob_mag], axis=-1)  # shape (Y,X,3)

    return feat


# ============================================================
# Step 5.1: Build features for ALL z-slices (stable version)
# ============================================================

Z = raw_crop.shape[0]
selected_z = np.arange(Z)   # use the entire crop

print("Selected z-slices:", selected_z)

# Allocate lists with fixed length = Z
X_slices = [None] * Z
y_slices = [None] * Z

# Build 3D features for every z-slice
for i, z in enumerate(selected_z):
    feat = extract_3d_features(raw_crop, z)  # (Y,X,3)
    lab  = label_multi[z]                   # (Y,X)

    # Flatten per-slice for RF training
    X_slices[i] = feat.reshape(-1, feat.shape[-1])
    y_slices[i] = lab.reshape(-1)


# ============================================================
# Step 6: K-fold Cross Validation (按 z 层划分)
# ============================================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_scores = []

# ============================================================
# Step 7: Train & Validate (selected_z only) — Updated Version
# ============================================================

import time
from tqdm import tqdm
from sklearn.metrics import jaccard_score, f1_score

fold_scores = []

labels_eval = [1, 2, 3, 4, 5, 6]   # evaluate only the six nucleus classes

# Outer progress bar for K-fold
for fold, (train_idx, val_idx) in tqdm(
    enumerate(kf.split(range(len(selected_z)))),
    total=kf.n_splits,
    desc="K-Fold Progress"
):

    print(f"\n================ FOLD {fold} ================")
    fold_start = time.time()

    # Build slice index lists
    train_z = [selected_z[i] for i in train_idx]
    val_z   = [selected_z[i] for i in val_idx]

    print("Train z slices:", train_z)
    print("Valid z slices:", val_z)

    # Inner progress bar (4 steps)
    fold_bar = tqdm(total=4, desc=f"Fold {fold} steps", leave=False)

    # ============================================================
    # Step A: Build training set
    # ============================================================
    fold_bar.set_postfix_str("Building train set")
    X_train = np.concatenate([X_slices[z] for z in train_z])
    y_train = np.concatenate([y_slices[z] for z in train_z])
    fold_bar.update(1)

    # ============================================================
    # Step B: Build validation set
    # ============================================================
    fold_bar.set_postfix_str("Building val set")
    X_val = np.concatenate([X_slices[z] for z in val_z])
    y_val = np.concatenate([y_slices[z] for z in val_z])
    fold_bar.update(1)

    print("Train size:", X_train.shape)
    print("Val size:  ", X_val.shape)

    # ============================================================
    # Step C: Train RandomForest
    # ============================================================
    fold_bar.set_postfix_str("Training RandomForest (may take minutes...)")
    clf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )
    clf.fit(X_train, y_train)
    fold_bar.update(1)

    # ============================================================
    # Step D: Predict validation set
    # ============================================================
    fold_bar.set_postfix_str("Predicting on val set")
    y_pred = clf.predict(X_val)
    fold_bar.update(1)
    fold_bar.close()

    # ============================================================
    # Step E: Compute evaluation metrics
    # ============================================================

    # Macro IoU & Dice only on 6 nucleus classes
    iou_macro = jaccard_score(y_val, y_pred, average="macro", labels=labels_eval)
    dice_macro = f1_score(y_val, y_pred, average="macro", labels=labels_eval)

    print(f"FOLD {fold}  IoU_macro={iou_macro:.4f}   Dice_macro={dice_macro:.4f}")

    # Per-class IoU
    per_class_iou = jaccard_score(
        y_val, y_pred,
        average=None,
        labels=labels_eval
    )

    print("Per-class IoU:")
    for cname, cid, ciou in zip(NUCLEUS_CLASSES, labels_eval, per_class_iou):
        print(f"  Class {cid:2d} ({cname:7s}) IoU = {ciou:.4f}")

    # Fold time
    elapsed = time.time() - fold_start
    print(f"⏱ Fold {fold} took {elapsed/60:.2f} minutes")

    fold_scores.append((iou_macro, dice_macro))


# ============================================================
# Step 8: Print final aggregated results
# ============================================================
print("\n================ Final K-fold Results ================\n")

for k, (iou, dice) in enumerate(fold_scores):
    print(f"Fold {k}:   IoU_macro={iou:.4f}   Dice_macro={dice:.4f}")

avg_iou  = np.mean([s[0] for s in fold_scores])
avg_dice = np.mean([s[1] for s in fold_scores])

print("\nAverage IoU (6 classes) :", avg_iou)
print("Average Dice (6 classes):", avg_dice)


# ============================================================
# Step 9: Save random 10 prediction slices into subfolder
# ============================================================

import random
from datetime import datetime

# ---- Create unique output folder ----
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_folder = f"../Result/pred_10slices_{timestamp}"
os.makedirs(save_folder, exist_ok=True)

print(f"\nSaving 10 prediction images into:\n{save_folder}")

# ---- Select 10 random z slices ----
num_vis = 10
vis_z_list = random.sample(range(Z), num_vis)
print("Selected slices:", vis_z_list)

for vis_z in vis_z_list:
    # Extract features
    feat_vis = extract_3d_features(raw_crop, vis_z)
    X_vis = feat_vis.reshape(-1, feat_vis.shape[-1])

    # Predict
    y_pred_vis = clf.predict(X_vis).reshape(Dy, Dx)

    # GT & raw
    y_gt_vis = label_multi[vis_z]
    raw_vis = raw_crop[vis_z]

    # ---- Plot ----
    fig = plt.figure(figsize=(18,6))

    plt.suptitle(f"Slice z={vis_z}", fontsize=16)

    plt.subplot(1,3,1)
    plt.title("Raw")
    plt.imshow(raw_vis, cmap='gray')
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("Ground Truth")
    plt.imshow(y_gt_vis, cmap="tab10")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title("Prediction")
    plt.imshow(y_pred_vis, cmap="tab10")
    plt.axis("off")

    # Fix title cutoff
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    # Save image
    save_path = os.path.join(save_folder, f"pred_slice_{vis_z}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved: {save_path}")
