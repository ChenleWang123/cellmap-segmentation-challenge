# =============================================
#   CellMap nucleus-related segmentation (crop292 only)
#   Classes: 20,21,22,23,24,28
#   Author: Chenle Wang
#   TU Dresden - CMS Research Project
# =============================================

import os
import json
import numpy as np
import time
import zarr # type: ignore
from tqdm import tqdm # type: ignore
import matplotlib.pyplot as plt # type: ignore
from scipy.ndimage import gaussian_filter, sobel, laplace # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import jaccard_score, f1_score # type: ignore
from sklearn.model_selection import KFold  # type: ignore
from scipy.ndimage import gaussian_filter, sobel, laplace # type: ignore


# ---------------------------------------------
# Step 1. Setup paths
# ---------------------------------------------

RAW_S0 = r"../data/jrc_cos7-1a/jrc_cos7-1a.zarr/recon-1/em/fibsem-uint8/s0"
CROP292_ROOT = r"../data/jrc_cos7-1a/jrc_cos7-1a.zarr/recon-1/labels/groundtruth/crop292"

# 5 classes
SELECT_CLASSES = {
    "mito_mem": 3,
    "mito_lum": 4,
    "hchrom": 24,
    "nucpl": 28,
    "cyto": 35
}

# multi-class mapping：background=0
CLASS_ID_MAP = {
    "mito_mem": 1,
    "mito_lum": 2,
    "hchrom": 3,
    "nucpl": 4,
    "cyto": 5
}

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
# Step 4. Build multi-class label (0–5) + visualization
# ---------------------------------------------

label_multi = np.zeros(ref_zarr.shape, dtype=np.uint8)

print("\n===== Loading 5-class labels =====")
for cname, real_id in SELECT_CLASSES.items():
    path = os.path.join(CROP292_ROOT, cname, "s0")
    arr = zarr.open(path, mode="r")[:]
    cid = CLASS_ID_MAP[cname]
    print(f"{cname:7s} -> class {cid}, unique={np.unique(arr)}")
    label_multi[arr > 0] = cid

print("Unique final multi-class labels:", np.unique(label_multi))

# ============================================================
# Step 5 (NEW): Full 3D Feature Extraction
# ============================================================
print("\n===== Performing Full 3D Feature Extraction =====")

def extract_full_3d_features(vol_3d):
    """
    Applies filters to the entire 3D volume, mimicking the method shown in the reference image.
    Input: (Z, Y, X) uint8 raw volume
    Output: (Z, Y, X, C) float32 feature volume
    """
    print(f"Input volume shape: {vol_3d.shape}. Extracting features...")
    start_time = time.time()
    
    # 1. Normalize to 0-1 range and convert to float32 (Crucial for many filters)
    image_vol = vol_3d.astype(np.float32) / 255.0
    
    features = []
    
    # --- Base Feature: Raw Intensity ---
    features.append(image_vol)
    
    # --- Multi-scale Gaussian Blur (To capture objects of different sizes) ---
    # Sigma is applied across all three axes (Z, Y, X)
    print("Applying 3D Gaussians...")
    features.append(gaussian_filter(image_vol, sigma=1.0))
    features.append(gaussian_filter(image_vol, sigma=2.0))
    features.append(gaussian_filter(image_vol, sigma=4.0))
    # Note: For anisotropic data (Z resolution different from XY), use a tuple, e.g., sigma=(0.5, 2.0, 2.0)
    
    # --- Texture and Edges ---
    print("Applying 3D Sobel & Laplace...")
    # 3D Sobel Magnitude
    sz = sobel(image_vol, axis=0)
    sy = sobel(image_vol, axis=1)
    sx = sobel(image_vol, axis=2)
    sob_mag = np.sqrt(sx**2 + sy**2 + sz**2)
    features.append(sob_mag)
    
    # Laplace (Second derivative, detects blobs and edges)
    features.append(laplace(image_vol))
    
    # Difference of Gaussians (DoG) - Effective for cell boundaries
    # Mimicking the formula: gaussian(..., sigma=1) - gaussian(..., sigma=3)
    dog_1_3 = gaussian_filter(image_vol, sigma=1.0) - gaussian_filter(image_vol, sigma=3.0)
    features.append(dog_1_3)

    # --- Stack Features ---
    # Stack along the last axis to form (Z, Y, X, N_features)
    stack = np.stack(features, axis=-1)
    
    elapsed = time.time() - start_time
    print(f"Features extracted in {elapsed:.2f}s. Result shape: {stack.shape}")
    return stack

# Execute feature extraction
# X_vol_4d shape: (Dz, Dy, Dx, n_features)
X_vol_4d = extract_full_3d_features(raw_crop)
# y_vol_3d shape: (Dz, Dy, Dx)
y_vol_3d = label_multi

Dz, Dy, Dx, n_feats = X_vol_4d.shape
print(f"Total features per voxel: {n_feats}")


# ============================================================
# NEW: Function Definition for Pixel Sampling (GLOBAL SCOPE)
# ============================================================

def limit_class_pixels(X, y, max_per_class=50000, random_state=42):
    """
    Limit the number of pixels per class to avoid memory explosion.
    """
    np.random.seed(random_state)
    X_list, y_list = [], []

    unique_classes = np.unique(y)
    print("\n===== Limiting class sizes =====")

    for c in unique_classes:
        idx = np.where(y == c)[0]
        count = len(idx)
        print(f"Class {c}: total {count} pixels")

        if count > max_per_class:
            selected = np.random.choice(idx, max_per_class, replace=False)
            print(f"   → sampled {max_per_class}")
        else:
            selected = idx
            print(f"   → kept all {count}")

        X_list.append(X[selected])
        y_list.append(y[selected])

    X_new = np.concatenate(X_list, axis=0)
    y_new = np.concatenate(y_list, axis=0)

    print(f"\nBalanced dataset size: {X_new.shape[0]} pixels\n")
    return X_new, y_new



# ============================================================
# Step 6: K-fold Cross Validation Setup
# ============================================================
# We still split based on Z slices to maintain validation independence
Z_indices = np.arange(Dz)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []
labels_eval = [1, 2, 3, 4, 5]

# ============================================================
# Step 7: Train & Validate (Applying Sampling to Train Set)
# ============================================================

for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(Z_indices)), total=kf.n_splits, desc="K-Fold Progress"):
    print(f"\n================ FOLD {fold} ================")
    fold_start = time.time()

    # ============================================================
    # Step A & B: Slicing and Reshaping Full Sets for the Fold
    # ============================================================
    print("Slicing and reshaping data for Random Forest...")

    # 1. Slice the sub-volumes based on Z indices (full data for this fold)
    # X_train_vol shape: (N_train_z, Dy, Dx, n_feats)
    X_train_vol = X_vol_4d[train_idx, :, :, :]
    y_train_vol = y_vol_3d[train_idx, :, :]
    
    X_val_vol = X_vol_4d[val_idx, :, :, :]
    y_val_vol = y_vol_3d[val_idx, :, :]
    
    # 2. Reshape full set for this fold
    X_train_full = X_train_vol.reshape(-1, n_feats)
    y_train_full = y_train_vol.reshape(-1)
    
    # Validation set remains full for accurate testing (no sampling)
    X_val = X_val_vol.reshape(-1, n_feats)
    y_val = y_val_vol.reshape(-1)

    # ------------------------------------------------------------
    # NEW: Apply pixel limit to the training set (Sampling)
    # ------------------------------------------------------------
    print(f"FOLD {fold}: Applying pixel limit of 50000 per class to training data...")
    
    # Use a unique random_state for each fold to ensure different, but reproducible, samples
    X_train, y_train = limit_class_pixels(
        X_train_full, 
        y_train_full, 
        max_per_class=50000, 
        random_state=42 + fold
    )
    # ------------------------------------------------------------
    
    print(f"Fold {fold} Sampled Train set shape:", X_train.shape)
    print(f"Fold {fold} Val set shape:  ", X_val.shape)

    # ============================================================
    # Step C: Train RandomForest 🌳
    # ============================================================
    print("Training RandomForest...")
    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)

    # ============================================================
    # Step D & E: Predict & Evaluate (No change)
    # ============================================================
    print("Predicting...")
    y_pred = clf.predict(X_val)
    
    # Macro IoU & Dice only on 5 nucleus classes
    iou_macro = jaccard_score(y_val, y_pred, average="macro", labels=labels_eval)
    dice_macro = f1_score(y_val, y_pred, average="macro", labels=labels_eval)
    print(f"FOLD {fold} Result: IoU={iou_macro:.4f}, Dice={dice_macro:.4f}")
    fold_scores.append((iou_macro, dice_macro))
    
    elapsed = time.time() - fold_start
    print(f"⏱ Fold {fold} took {elapsed/60:.2f} minutes")



# ============================================================
# Step 7.5 (NEW): Train final model on ALL data (Balanced)
# ============================================================

print("\n===== Step 7.5: Training final model on ALL data (Balanced Sampling) =====")

# ------------------------------------------------------------
# A. Flatten full volume
# ------------------------------------------------------------
print("Reshaping full 4D feature volume...")
X_all = X_vol_4d.reshape(-1, n_feats)   # shape: (Dz*Dy*Dx, n_features)
y_all = y_vol_3d.reshape(-1)            # shape: (Dz*Dy*Dx,)

print(f"Full dataset shape: X={X_all.shape}, y={y_all.shape}")


# Apply the sampling
X_sub, y_sub = limit_class_pixels(X_all, y_all, max_per_class=50000)


# ------------------------------------------------------------
# C. Train final RandomForest
# ------------------------------------------------------------
print("Training FINAL RandomForest model...")

final_clf = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
    max_depth=None,        # 可根据需要调整
)

final_clf.fit(X_sub, y_sub)

print("🎉 Final model training complete!")



# # --- NEW: Save the final trained model ---
# model_filename = os.path.join(save_folder, "final_rf_model.pkl")
# print(f"Saving final model to: {model_filename}")
# joblib.dump(final_clf, model_filename)
# print("Model saved successfully!")


# ============================================================
# Step 8: Print final aggregated results
# ============================================================
print("\n================ Final K-fold Results ================\n")

for k, (iou, dice) in enumerate(fold_scores):
    print(f"Fold {k}:   IoU_macro={iou:.4f}   Dice_macro={dice:.4f}")

avg_iou  = np.mean([s[0] for s in fold_scores])
avg_dice = np.mean([s[1] for s in fold_scores])

print("\nAverage IoU (5 classes) :", avg_iou)
print("Average Dice (5 classes):", avg_dice)



# ============================================================
# Step 9: Save random 10 prediction slices into subfolder (Updated)
# ============================================================

import random
from datetime import datetime

# ---- Create unique output folder ----
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_folder = f"../Result/pred_10slices_{timestamp}"
os.makedirs(save_folder, exist_ok=True)

print(f"\nSaving 10 prediction images into:\n{save_folder}")

# --- define Dy, Dx for reshape ---
# Dz, Dy, Dx were defined after Step 5 (X_vol_4d.shape)
# Dz, Dy, Dx = raw_crop.shape

# ---- Select 10 random z slices ----
num_vis = 10
vis_z_list = random.sample(range(Dz), num_vis) # Use Dz (total depth)
print("Selected slices:", vis_z_list)

for vis_z in vis_z_list:
    print(f"Visualizing slice z={vis_z}...")
    
    # NEW: Retrieve the pre-computed features for this Z slice
    # feat_vis_slice shape: (Dy, Dx, n_feats)
    feat_vis_slice = X_vol_4d[vis_z, :, :, :]
    
    # Reshape to (Dy*Dx, n_feats) for prediction
    X_vis = feat_vis_slice.reshape(-1, n_feats)

    # Predict full 2D segmentation map
    # The result is (Dy*Dx,), which is then reshaped back to image shape (Dy, Dx)
    y_pred_vis = final_clf.predict(X_vis).reshape(Dy, Dx)

    # GT & raw (retrieve original data)
    y_gt_vis = label_multi[vis_z]
    raw_vis  = raw_crop[vis_z]

    # ---- Visualization ----
    fig = plt.figure(figsize=(18, 6))

    plt.suptitle(f"Slice z={vis_z}", fontsize=16)

    # --- Raw ---
    plt.subplot(1, 3, 1)
    plt.title("Raw")
    plt.imshow(raw_vis, cmap='gray')
    plt.axis("off")

    # --- Ground Truth ---
    plt.subplot(1, 3, 2)
    plt.title("Ground Truth")
    plt.imshow(y_gt_vis, cmap="tab10")
    plt.axis("off")

    # --- Prediction ---
    plt.subplot(1, 3, 3)
    plt.title("Prediction")
    plt.imshow(y_pred_vis, cmap="tab10")
    plt.axis("off")

    # fix title cutoff
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    # Save image
    save_path = os.path.join(save_folder, f"pred_slice_{vis_z}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved: {save_path}")