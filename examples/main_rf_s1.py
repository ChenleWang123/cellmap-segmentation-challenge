# =============================================
#   CellMap nucleus-related segmentation (Multi-Crop K-Fold)
#   Classes: 20,21,22,23,24,28,35
#   Author: Chenle
#   TU Dresden - CMS Research Project
# =============================================

import os
import json
import numpy as np
import time
import zarr  # type: ignore
from tqdm import tqdm  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import random
from datetime import datetime
from scipy.ndimage import gaussian_filter, sobel, laplace  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.model_selection import KFold  # type: ignore
from sklearn.metrics import jaccard_score, f1_score  # type: ignore
from scipy.ndimage import (
    maximum_filter,
    minimum_filter,
    median_filter,
)  # 新增 maximum/minimum/median # type: ignore
from skimage.feature import (
    hessian_matrix,
    hessian_matrix_eigvals,
)  # 新增：需要安装 scikit-image # type: ignore

# ---------------------------------------------
# Step 1. Setup paths (MODIFIED)
# ---------------------------------------------

# Define all Crop IDs to be used in the K-Fold cross-validation (5 crops for 5-fold)
# KFold(n_splits=len(CROP_IDS)) will automatically set it to 9 folds.
# Each fold: 8 for training, 1 for validation.
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

RAW_S0 = r"../data/jrc_cos7-1a/jrc_cos7-1a.zarr/recon-1/em/fibsem-uint8/s1"
GROUNDTRUTH_ROOT = r"../data/jrc_cos7-1a/jrc_cos7-1a.zarr/recon-1/labels/groundtruth"

# 5 classes for segmentation
SELECT_CLASSES = {
    "cyto": 35,
    "mito_mem": 3,
    "mito_lum": 4,
    "er_mem": 16,
    "er_lum": 17,
}

# Multi-class mapping: background=0
CLASS_ID_MAP = {
    "cyto": 1,
    "mito_mem": 2,
    "mito_lum": 3,
    "er_mem": 4,
    "er_lum": 5,
}

# Class name list for printing results (order MUST match labels_eval)
CLASS_NAMES_ORDERED = ["cyto", "mito_mem", "mito_lum", "er_mem", "er_lum"]
labels_eval = [1, 2, 3, 4, 5]  # The 5 target class IDs

REF_CLASS = "nucpl"  # Used to determine the bounds/shape of each crop

raw_zarr = zarr.open(RAW_S0, mode="r")
print("Raw shape:", raw_zarr.shape)

# ============================================================
# NEW: Function Definition for 3D Feature Extraction (Step 5 logic)
# ============================================================


def extract_full_3d_features_40ch(vol_3d):
    """
    Updated Feature set: 32 Original + 8 New = 40 Channels
    """
    # 1. Normalize Input
    img = vol_3d.astype(np.float32) / 255.0
    features = []

    # Helper: Precompute Gaussians
    sigmas = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0]
    gaussians = {}

    # --- Group 1: Raw & Gaussians (9 channels) ---
    features.append(img)
    for s in sigmas:
        g = gaussian_filter(img, sigma=s)
        gaussians[s] = g
        features.append(g)

    # --- Group 2: Gradient Magnitude (5 channels) ---
    def get_grad_mag(smoothed_vol):
        gz = sobel(smoothed_vol, axis=0)
        gy = sobel(smoothed_vol, axis=1)
        gx = sobel(smoothed_vol, axis=2)
        return np.sqrt(gz**2 + gy**2 + gx**2)

    grad_sigmas = [1.0, 2.0, 4.0, 8.0, 12.0]
    for s in grad_sigmas:
        features.append(get_grad_mag(gaussians[s]))

    # --- Group 3: Laplacian of Gaussian (5 channels) ---
    log_sigmas = [1.0, 2.0, 4.0, 8.0, 12.0]
    for s in log_sigmas:
        features.append(laplace(gaussians[s]))

    # --- Group 4: DoG (5 channels) ---
    dog_pairs = [(1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 16.0)]
    for s1, s2 in dog_pairs:
        features.append(gaussians[s1] - gaussians[s2])

    # --- Group 5: Hessian Diagonal Approx (Old) (6 channels) ---
    # We keep this to preserve your original 32 structure,
    # even though Eigenvalues (below) are better.
    def get_hessian_diag(smoothed_vol):
        zz = sobel(sobel(smoothed_vol, axis=0), axis=0)
        yy = sobel(sobel(smoothed_vol, axis=1), axis=1)
        xx = sobel(sobel(smoothed_vol, axis=2), axis=2)
        return xx, yy, zz

    hxx, hyy, hzz = get_hessian_diag(gaussians[2.0])
    features.extend([hxx, hyy, hzz])
    hxx, hyy, hzz = get_hessian_diag(gaussians[4.0])
    features.extend([hxx, hyy, hzz])

    # --- Group 6: Local Variance (2 channels) ---
    def get_std(raw, smoothed, s):
        mean_sq = gaussian_filter(raw**2, sigma=s)
        mean = smoothed
        var = mean_sq - mean**2
        return np.sqrt(np.maximum(var, 0))

    features.append(get_std(img, gaussians[2.0], 2.0))
    features.append(get_std(img, gaussians[4.0], 4.0))

    # =========================================================
    # NEW FEATURES (8 Channels)
    # =========================================================

    # --- Group 7: Local Range / Texture (2 channels) ---
    # Calculates Max - Min in a neighborhood.
    # Great for distinguishing messy cytoplasm from smooth organelles.
    def get_range(vol, size):
        mx = maximum_filter(vol, size=size)
        mn = minimum_filter(vol, size=size)
        return mx - mn

    features.append(get_range(img, size=3))  # Fine texture (Ch 32)
    features.append(get_range(gaussians[1.0], size=5))  # Coarse texture (Ch 33)

    # --- Group 8: True Hessian Eigenvalues (6 channels) ---
    # Captures Shape: Sheet (Membrane) vs Tube vs Blob
    def get_eigenvalues(vol, sigma):
        # returns list [lambda1, lambda2, lambda3] per pixel
        # This uses skimage, which is optimized
        # H_elems = hessian_matrix(vol, sigma=sigma, order="rc")
        H_elems = hessian_matrix(vol, sigma=sigma, order="rc", use_gaussian_derivatives=False)
        eigvals = hessian_matrix_eigvals(H_elems)
        return eigvals  # list of 3 arrays

    # Scale 2.0 (Organelle Boundaries)
    eigs_s2 = get_eigenvalues(img, sigma=2.0)
    features.extend(eigs_s2)  # Ch 34, 35, 36

    # Scale 4.0 (Larger Context)
    eigs_s4 = get_eigenvalues(img, sigma=4.0)
    features.extend(eigs_s4)  # Ch 37, 38, 39

    # Stack
    stack = np.stack(features, axis=-1).astype(np.float32)

    # Safety Check
    if stack.shape[-1] != 40:
        print(f"Warning: Features count is {stack.shape[-1]}, expected 40.")
        if stack.shape[-1] > 40:
            stack = stack[..., :40]

    return stack


# ============================================================
# Step 2-4: Load multiple crops, align, and store (MODIFIED for s1)
# ============================================================

# List to store the loaded raw data, labels, and metadata for each crop
all_crops_data = []
crop_pixel_counts = []  # Stores total pixels (Dz * Dy * Dx) for each crop

# !!! define scale factor：s1 is 1/2 of s0 !!!
SCALE_FACTOR = 2

print("\n===== Loading and Aligning Multiple Crops (s1 Resolution) =====")

for crop_id in CROP_IDS:
    print(f"\nProcessing {crop_id}...")
    CROP_ROOT = os.path.join(GROUNDTRUTH_ROOT, crop_id)
    REF_S0 = os.path.join(CROP_ROOT, REF_CLASS, "s0")
    REF_ZATTR = os.path.join(CROP_ROOT, REF_CLASS, ".zattrs")

    # 1. Open reference zarr to get s0 shape (Ground Truth Shape)
    ref_zarr_s0 = zarr.open(REF_S0, mode="r")
    Dz_s0, Dy_s0, Dx_s0 = ref_zarr_s0.shape

    # --- Step 2. Read .zattrs (translation + scale based on s0) ---
    with open(REF_ZATTR, "r") as f:
        attrs = json.load(f)
    ms = attrs["multiscales"][0]["datasets"][0]
    scale = ms["coordinateTransformations"][0]["scale"]
    trans = ms["coordinateTransformations"][1]["translation"]
    scale_z, scale_y, scale_x = scale
    tz, ty, tx = trans

    # --- Step 3. Convert nm -> s0 voxel index -> s1 voxel index ---
    # a. Calculate s0 indices first
    vz0_s0 = int(tz / scale_z)
    vy0_s0 = int(ty / scale_y)
    vx0_s0 = int(tx / scale_x)

    # b. Convert to s1 indices (Divide by SCALE_FACTOR)
    vz0 = vz0_s0 // SCALE_FACTOR
    vy0 = vy0_s0 // SCALE_FACTOR
    vx0 = vx0_s0 // SCALE_FACTOR

    # c. Calculate s1 shape (Divide s0 shape by SCALE_FACTOR)
    Dz = Dz_s0 // SCALE_FACTOR
    Dy = Dy_s0 // SCALE_FACTOR
    Dx = Dx_s0 // SCALE_FACTOR

    # d. Calculate end indices for s1
    vz1, vy1, vx1 = vz0 + Dz, vy0 + Dy, vx0 + Dx

    # Load Raw Data (from s1 source, using s1 indices)
    # Assumes 'raw_zarr' variable points to the s1 zarr array
    raw_crop = raw_zarr[vz0:vz1, vy0:vy1, vx0:vx1]
    print(f"  Raw crop shape (s1): {raw_crop.shape}")

    # --- Step 4. Build multi-class label (s0 -> downsample -> s1) ---
    label_multi = np.zeros((Dz, Dy, Dx), dtype=np.uint8)
    for cname, real_id in SELECT_CLASSES.items():
        path = os.path.join(CROP_ROOT, cname, "s0")
        try:
            # Load s0 binary mask
            arr_s0 = zarr.open(path, mode="r")[:]
            # Downsample to s1: Slice every 2nd pixel [::2, ::2, ::2]
            arr_s1 = arr_s0[::SCALE_FACTOR, ::SCALE_FACTOR, ::SCALE_FACTOR]
            # Clip shape if mismatch (handle odd/even rounding issues)
            arr_s1 = arr_s1[:Dz, :Dy, :Dx]
            cid = CLASS_ID_MAP[cname]
            label_multi[arr_s1 > 0] = cid
        except Exception as e:
            print(f"  Warning: Failed to load class {cname} in {crop_id}. Error: {e}")

    # Safety Check: Ensure Raw and Label shapes match exactly
    if raw_crop.shape != label_multi.shape:
        print(
            f"  ! Shape Mismatch Fixed: Raw {raw_crop.shape} vs Label {label_multi.shape}"
        )
        min_z = min(raw_crop.shape[0], label_multi.shape[0])
        min_y = min(raw_crop.shape[1], label_multi.shape[1])
        min_x = min(raw_crop.shape[2], label_multi.shape[2])
        raw_crop = raw_crop[:min_z, :min_y, :min_x]
        label_multi = label_multi[:min_z, :min_y, :min_x]
        Dz, Dy, Dx = raw_crop.shape  # Update dimensions

    print(f"  Unique final multi-class labels in {crop_id}: {np.unique(label_multi)}")

    # Store the result and metadata
    all_crops_data.append(
        {"raw": raw_crop, "label": label_multi, "shape": (Dz, Dy, Dx), "id": crop_id}
    )
    crop_pixel_counts.append(Dz * Dy * Dx)

# ============================================================
# Step 5: (LIGHT) Determine feature count only (NO global merge!)
# ============================================================
print("\n===== Step 5: Determining Feature Count (No Global Merge) =====")

# Just compute n_feats once using the first crop
tmp_raw = all_crops_data[0]["raw"]
tmp_feat = extract_full_3d_features_40ch(tmp_raw)
n_feats = tmp_feat.shape[-1]
del tmp_feat
print(f"Total features per voxel: {n_feats}")


# ============================================================
# NEW: Function Definition for Pixel Sampling
# ============================================================


def limit_class_pixels(X, y, max_per_class=50000, random_state=42):
    """
    Limit the number of pixels per class to avoid memory explosion and mitigate imbalance.
    """
    np.random.seed(random_state)
    X_list, y_list = [], []
    all_unique_classes = np.unique(y)
    print("\n===== Limiting class sizes =====")

    for c in all_unique_classes:
        idx = np.where(y == c)[0]
        count = len(idx)
        cname = "Background" if c == 0 else CLASS_NAMES_ORDERED[c - 1]
        print(f"Class {c} ({cname}): total {count} pixels")
        if count > max_per_class:
            selected = np.random.choice(idx, max_per_class, replace=False)
            print(f"   -> sampled {max_per_class}")
        else:
            selected = idx
            print(f"   -> kept all {count}")
        X_list.append(X[selected])
        y_list.append(y[selected])

    X_new = np.concatenate(X_list, axis=0)
    y_new = np.concatenate(y_list, axis=0)
    print(f"\nBalanced dataset size: {X_new.shape[0]} pixels")
    return X_new, y_new

def sample_from_crop(X_flat, y_flat, max_per_class=50000, random_state=42):
    """
    Sample up to max_per_class pixels PER CLASS from ONE crop.
    Returns a much smaller (X_s, y_s) to append into the fold training set.
    """
    rng = np.random.RandomState(random_state)
    X_list, y_list = [], []

    for c in np.unique(y_flat):
        idx = np.where(y_flat == c)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, max_per_class, replace=False)
        X_list.append(X_flat[idx])
        y_list.append(y_flat[idx])

    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


# ============================================================
# Step 6: K-fold Cross Validation Setup (MODIFIED)
# ============================================================
# K-Fold split is based on the index of the crops
crop_indices = np.arange(len(CROP_IDS))
kf = KFold(n_splits=len(CROP_IDS), shuffle=True, random_state=42)

# Calculate cumulative pixel count indices to slice X_all_flat and y_all_flat
# This array tells us where each crop starts in the flattened dataset
# crop_start_indices = np.cumsum([0] + crop_pixel_counts)
fold_scores = []

# ============================================================
# Step 7: Train & Validate (MEMORY-SAFE VERSION)
# ============================================================

for fold, (train_crop_idx, val_crop_idx) in tqdm(
    enumerate(kf.split(crop_indices)), total=kf.n_splits, desc="K-Fold Progress"
):
    print(f"\n================ FOLD {fold} ================")
    fold_start = time.time()

    val_c_idx = val_crop_idx[0]
    print(f"FOLD {fold}: Validation Crop ID: {CROP_IDS[val_c_idx]}")

    # ------------------------------------------------------------
    # A. Build TRAIN set by: feature extraction per crop -> sampling per crop -> concat
    # ------------------------------------------------------------
    PIXEL_LIMIT = 50000  # per class
    print(f"FOLD {fold}: Building TRAIN set with per-crop sampling limit {PIXEL_LIMIT} per class...")

    X_train_parts, y_train_parts = [], []

    for c_idx in train_crop_idx:
        crop_id = all_crops_data[c_idx]["id"]
        raw_crop = all_crops_data[c_idx]["raw"]
        label_multi = all_crops_data[c_idx]["label"]

        print(f"  [Train] Extracting features for {crop_id}...")
        X_vol_4d = extract_full_3d_features_40ch(raw_crop)
        X_flat = X_vol_4d.reshape(-1, n_feats)
        y_flat = label_multi.reshape(-1)

        # sample immediately (avoid huge X_train_full)
        X_s, y_s = sample_from_crop(
            X_flat, y_flat, max_per_class=PIXEL_LIMIT, random_state=42 + fold + c_idx
        )

        X_train_parts.append(X_s)
        y_train_parts.append(y_s)

        # free big arrays ASAP
        del X_vol_4d, X_flat, y_flat, X_s, y_s

    X_train = np.concatenate(X_train_parts, axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)
    del X_train_parts, y_train_parts

    print(f"Fold {fold} Sampled Train set shape: {X_train.shape}")

    # ------------------------------------------------------------
    # B. Build VAL set: full validation crop features (chunked predict later)
    # ------------------------------------------------------------
    val_crop_data = all_crops_data[val_c_idx]
    Dz_val, Dy_val, Dx_val = val_crop_data["shape"]

    print(f"  [Val] Extracting features for {val_crop_data['id']}...")
    X_val_4d = extract_full_3d_features_40ch(val_crop_data["raw"])
    X_val = X_val_4d.reshape(-1, n_feats)
    y_val = val_crop_data["label"].reshape(-1)
    del X_val_4d

    print(f"Fold {fold} Val set shape:  {X_val.shape}")

    # ============================================================
    # Step C: Train RandomForest 🌳
    # ============================================================
    print("Training RandomForest...")
    clf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", n_jobs=2, random_state=42
    )
    clf.fit(X_train, y_train)

    # free train arrays (RF already fitted)
    del X_train, y_train

    # ============================================================
    # Step D & E: Predict & Evaluate (USING CHUNKING)
    # ============================================================
    print("Predicting...")
    chunk_size = 2000000  # ⬅️ 建议比你原来的 8e6 小一些，更稳
    y_pred_list = []
    n_pixels = X_val.shape[0]
    n_chunks = int(np.ceil(n_pixels / chunk_size))

    for i in tqdm(range(n_chunks), desc="Predicting Chunks"):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_pixels)
        X_chunk = X_val[start_idx:end_idx]
        y_pred_chunk = clf.predict(X_chunk)
        y_pred_list.append(y_pred_chunk)

    y_pred_flat = np.concatenate(y_pred_list, axis=0)
    del y_pred_list, X_val  # free val features ASAP

    # reshape to 3D for median post-processing
    y_pred_3d = y_pred_flat.reshape(Dz_val, Dy_val, Dx_val)
    del y_pred_flat

    print("Applying Post-processing (Median Filter)...")
    y_pred_3d_refined = median_filter(y_pred_3d, size=3)
    del y_pred_3d

    y_pred_final = y_pred_3d_refined.reshape(-1)
    del y_pred_3d_refined

    print(f"Prediction complete. Final shape: {y_pred_final.shape}")

    iou_per_class = jaccard_score(
        y_val, y_pred_final, average=None, labels=labels_eval, zero_division=0
    )
    dice_per_class = f1_score(
        y_val, y_pred_final, average=None, labels=labels_eval, zero_division=0
    )
    iou_macro = np.mean(iou_per_class)
    dice_macro = np.mean(dice_per_class)
    print(f"FOLD {fold} Result: IoU_macro={iou_macro:.4f}, Dice_macro={dice_macro:.4f}")

    fold_scores.append(
        {
            "iou_macro": iou_macro,
            "dice_macro": dice_macro,
            "iou_per_class": iou_per_class,
            "dice_per_class": dice_per_class,
        }
    )

    elapsed = time.time() - fold_start
    print(f"⏱ Fold {fold} took {elapsed / 60:.2f} minutes")

    # free remaining big arrays
    del y_val, y_pred_final, clf


# ============================================================
# Step 7.5: Train final model on ALL crops (Memory-safe sampling)
# ============================================================

print("\n===== Step 7.5: Training final model on ALL data (Balanced Sampling) =====")

PIXEL_LIMIT_FINAL = 50000
X_parts, y_parts = [], []

for idx, crop_item in enumerate(all_crops_data):
    print(f"  [Final] Extracting + sampling from {crop_item['id']}...")
    X_vol_4d = extract_full_3d_features_40ch(crop_item["raw"])
    X_flat = X_vol_4d.reshape(-1, n_feats)
    y_flat = crop_item["label"].reshape(-1)

    X_s, y_s = sample_from_crop(
        X_flat, y_flat, max_per_class=PIXEL_LIMIT_FINAL, random_state=123 + idx
    )
    X_parts.append(X_s)
    y_parts.append(y_s)

    del X_vol_4d, X_flat, y_flat, X_s, y_s

X_sub = np.concatenate(X_parts, axis=0)
y_sub = np.concatenate(y_parts, axis=0)
del X_parts, y_parts

print("Training FINAL RandomForest model...")

final_clf = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    n_jobs=2,
    random_state=42,
)
final_clf.fit(X_sub, y_sub)

del X_sub, y_sub
print("🎉 Final model training complete!")


# ============================================================
# Step 8: Print final aggregated results (NO CHANGE)
# ============================================================
print("\n================ Final K-fold Results ================\n")

# Aggregate per-class scores across all folds
all_iou_per_class = np.array([s["iou_per_class"] for s in fold_scores])
all_dice_per_class = np.array([s["dice_per_class"] for s in fold_scores])

# Calculate average per-class scores
avg_iou_per_class = np.mean(all_iou_per_class, axis=0)
avg_dice_per_class = np.mean(all_dice_per_class, axis=0)

# Print detailed results per fold
for k, scores in enumerate(fold_scores):
    print(
        f"Fold {k}:   IoU_macro={scores['iou_macro']:.4f}   Dice_macro={scores['dice_macro']:.4f}"
    )

print("\n---------------- Detailed Class Averages ----------------")
print("Class | Avg IoU (Jaccard) | Avg Dice (F1)")
print("------|-------------------|---------------")
for i, cname in enumerate(CLASS_NAMES_ORDERED):
    print(f"{cname:6s}| {avg_iou_per_class[i]:.4f} | {avg_dice_per_class[i]:.4f}")

    # Calculate final macro averages
avg_macro_iou = np.mean(avg_iou_per_class)
avg_macro_dice = np.mean(avg_dice_per_class)

print("\n================ Aggregated Macro Scores ================")
print(f"Average Macro IoU (5 classes): {avg_macro_iou:.4f}")
print(f"Average Macro Dice (5 classes): {avg_macro_dice:.4f}")

# ============================================================
# Step 9: Save 10 random prediction slices per crop (MODIFIED with Sorted Legend)
# ============================================================

# ---- 1. Determine the next experiment number (i) ----
result_root = "../Result"
os.makedirs(result_root, exist_ok=True)
existing_dirs = [d for d in os.listdir(result_root) if d.startswith("predict")]
max_i = 0
for d in existing_dirs:
    try:
        num_str = "".join(filter(str.isdigit, d))
        if num_str:
            max_i = max(max_i, int(num_str))
    except:
        continue

next_i = max_i + 1
experiment_folder = os.path.join(result_root, f"predict({next_i})")
os.makedirs(experiment_folder, exist_ok=True)

print(f"\nSaving visualizations into new experiment folder: {experiment_folder}")

# ---- 2. Prepare Legend Data (Matches your SELECT_CLASSES IDs) ----
# Same logic as before: Bind colors to internal indices (0-5) but show Real IDs
cmap = plt.get_cmap("tab10")
# Internal names matching the class indices
CLASS_NAMES = ["bg", "cyto", "mito_mem", "mito_lum", "er_mem", "er_lum"]
# Original ID mapping for display
SELECT_CLASSES_MAP = {
    "cyto": 35,
    "mito_mem": 3,
    "mito_lum": 4,
    "er_mem": 16,
    "er_lum": 17,
}

legend_items = []
for idx, name in enumerate(CLASS_NAMES):
    real_id = 0 if name == "bg" else SELECT_CLASSES_MAP[name]
    legend_items.append(
        {
            "model_idx": idx,
            "real_id": real_id,
            "label": f"{name} ({real_id})",
            "color": cmap(idx),
        }
    )

# Sort legend items by Real ID for a professional look
legend_items.sort(key=lambda x: x["real_id"])
legend_patches = [
    plt.Line2D(
        [0],
        [0],
        color=item["color"],
        lw=4,
        label=item["label"],
        marker="s",
        linestyle="None",
        markersize=10,
    )
    for item in legend_items
]

# ---- 3. Iterate through all crops and save slices ----
num_vis_per_crop = 5
crops_data_dict = {item["id"]: item for item in all_crops_data}

if "final_clf" not in locals():
    print("Warning: final_clf not found. Skipping visualization.")
else:
    for crop_id in CROP_IDS:
        crop_item = crops_data_dict[crop_id]
        crop_save_folder = os.path.join(experiment_folder, crop_id)
        os.makedirs(crop_save_folder, exist_ok=True)
        print(f"\n--- Processing {crop_id} (Saving {num_vis_per_crop} slices) ---")

        raw_crop_vis = crop_item["raw"]
        label_multi_vis = crop_item["label"]
        Dz, Dy, Dx = crop_item["shape"]
        # feat_vol_4d = crop_item["features_4d"]
        feat_vol_4d = extract_full_3d_features_40ch(raw_crop_vis)

        # Select random Z slices
        if Dz < num_vis_per_crop:
            vis_zs = range(Dz)
        else:
            # Generate evenly spaced indices from 0 to Dz-1
            vis_zs = np.linspace(0, Dz - 1, num=num_vis_per_crop, dtype=int).tolist()
            # Sort and remove duplicates (just in case)
            vis_zs = sorted(list(set(vis_zs)))
        # else:
        #     vis_zs = random.sample(range(Dz), num_vis_per_crop)

        for i, vis_z in enumerate(vis_zs):
            # Predict using the classifier on the selected slice
            feat_vis_slice = feat_vol_4d[vis_z, :, :, :]
            X_vis = feat_vis_slice.reshape(-1, n_feats)
            y_pred_vis = final_clf.predict(X_vis).reshape(Dy, Dx)

            # Apply the same post-processing to the visualization to match your metrics
            y_pred_vis = median_filter(y_pred_vis, size=3)
            # --------------------------------------

            y_gt_vis = label_multi_vis[vis_z]
            raw_vis = raw_crop_vis[vis_z]

            # ---- Start Visualization ----
            # Increase figure size to accommodate the legend on the right
            fig = plt.figure(figsize=(20, 6))
            plt.suptitle(
                f"Exp: {next_i} | Crop: {crop_id}, Slice z={vis_z}", fontsize=16
            )

            # --- 1. Raw ---
            plt.subplot(1, 3, 1)
            plt.title("Raw")
            plt.imshow(raw_vis, cmap="gray")
            plt.axis("off")

            # --- 2. Ground Truth (GT) ---
            plt.subplot(1, 3, 2)
            plt.title("Ground Truth")
            plt.imshow(y_gt_vis, cmap=cmap, vmin=0, vmax=9)
            plt.axis("off")

            # Add Legend to the right of GT
            plt.legend(
                handles=legend_patches,
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                title="Classes (ID)",
                borderaxespad=0.0,
            )

            # --- 3. Prediction ---
            plt.subplot(1, 3, 3)
            plt.title("Prediction")
            plt.imshow(y_pred_vis, cmap=cmap, vmin=0, vmax=9)
            plt.axis("off")

            # Add Legend to the right of Prediction
            plt.legend(
                handles=legend_patches,
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                title="Classes (ID)",
                borderaxespad=0.0,
            )

            plt.tight_layout(rect=[0, 0, 1, 0.94])
            # Save image
            save_path = os.path.join(
                crop_save_folder, f"slice_{i + 1:02d}_z{vis_z:04d}.png"
            )
            plt.savefig(save_path, dpi=200)
            plt.close()
            print(f"  [{i + 1}/{len(vis_zs)}] Saved: {save_path}")

    print("\n--- Script Finished ---")
