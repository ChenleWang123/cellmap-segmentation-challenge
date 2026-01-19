# =============================================
#   CellMap nucleus-related segmentation (Multi-Crop K-Fold)
#   Classes: 20,21,22,23,24,28,35
#   Author: Chenle
#   TU Dresden - CMS Research Project
#
#   UPDATED (2026-01-18):
#   1) CROP_IDS updated to 9 crops (9-fold)
#   2) Feature extraction strictly 40 channels
#   3) Output path format:
#      /home/chwa386g/chwa386g/cellmap-segmentation-challenge/Result/RF/run_YYYYMMDD_HHMMSS/fold_XX/<val_crop>/*.png
#      - run timestamp generated ONCE per run
#      - per-fold saves ONLY the validation crop slices
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
)  # type: ignore
from skimage.feature import (
    hessian_matrix,
    hessian_matrix_eigvals,
)  # type: ignore

# ---------------------------------------------
# Step 1. Setup paths (MODIFIED)
# ---------------------------------------------

# ✅ Updated Crop IDs (9 crops for 9-fold)
CROP_IDS = ["crop292", "crop234", "crop236", "crop237", "crop239"]

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
# NEW: One-time RUN folder (auto timestamp, only once)
# ============================================================
RUN_ROOT = "/home/chwa386g/chwa386g/cellmap-segmentation-challenge/Result/RF"
run_stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(RUN_ROOT, run_stamp)
os.makedirs(RUN_DIR, exist_ok=True)
print(f"\n[RUN] Saving all fold visualizations into: {RUN_DIR}")

# ============================================================
# NEW: Prepare legend once (before KFold loop)
# ============================================================
cmap = plt.get_cmap("tab10")
CLASS_NAMES = ["bg", "cyto", "mito_mem", "mito_lum", "er_mem", "er_lum"]
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
        {"model_idx": idx, "real_id": real_id, "label": f"{name} ({real_id})", "color": cmap(idx)}
    )

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

# ============================================================
# NEW: Function Definition for 3D Feature Extraction (40 Channels)
# ============================================================


def extract_full_3d_features_40ch(vol_3d):
    """
    Feature set: 40 channels
    32 original + 8 new (range + Hessian eigenvalues).
    STRICT: must be exactly 40, else raise.
    """
    img = vol_3d.astype(np.float32) / 255.0
    features = []

    sigmas = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0]
    gaussians = {}

    # --- Group 1: Raw & Gaussians (1 + 8 = 9) ---
    features.append(img)
    for s in sigmas:
        g = gaussian_filter(img, sigma=s)
        gaussians[s] = g
        features.append(g)

    # --- Group 2: Gradient Magnitude (5) ---
    def get_grad_mag(smoothed_vol):
        gz = sobel(smoothed_vol, axis=0)
        gy = sobel(smoothed_vol, axis=1)
        gx = sobel(smoothed_vol, axis=2)
        return np.sqrt(gz**2 + gy**2 + gx**2)

    grad_sigmas = [1.0, 2.0, 4.0, 8.0, 12.0]
    for s in grad_sigmas:
        features.append(get_grad_mag(gaussians[s]))

    # --- Group 3: Laplacian of Gaussian (5) ---
    log_sigmas = [1.0, 2.0, 4.0, 8.0, 12.0]
    for s in log_sigmas:
        features.append(laplace(gaussians[s]))

    # --- Group 4: DoG (5) ---
    dog_pairs = [(1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 16.0)]
    for s1, s2 in dog_pairs:
        features.append(gaussians[s1] - gaussians[s2])

    # --- Group 5: Hessian Diagonal Approx (Old) (6) ---
    def get_hessian_diag(smoothed_vol):
        zz = sobel(sobel(smoothed_vol, axis=0), axis=0)
        yy = sobel(sobel(smoothed_vol, axis=1), axis=1)
        xx = sobel(sobel(smoothed_vol, axis=2), axis=2)
        return xx, yy, zz

    hxx, hyy, hzz = get_hessian_diag(gaussians[2.0])
    features.extend([hxx, hyy, hzz])
    hxx, hyy, hzz = get_hessian_diag(gaussians[4.0])
    features.extend([hxx, hyy, hzz])

    # --- Group 6: Local Variance (2) ---
    def get_std(raw, smoothed, s):
        mean_sq = gaussian_filter(raw**2, sigma=s)
        mean = smoothed
        var = mean_sq - mean**2
        return np.sqrt(np.maximum(var, 0))

    features.append(get_std(img, gaussians[2.0], 2.0))
    features.append(get_std(img, gaussians[4.0], 4.0))

    # --- Group 7: Local Range / Texture (2) ---
    def get_range(vol, size):
        mx = maximum_filter(vol, size=size)
        mn = minimum_filter(vol, size=size)
        return mx - mn

    features.append(get_range(img, size=3))           # Ch 32
    features.append(get_range(gaussians[1.0], size=5))  # Ch 33

    # --- Group 8: True Hessian Eigenvalues (6) ---
    def get_eigenvalues(vol, sigma):
        H_elems = hessian_matrix(vol, sigma=sigma, order="rc")
        eigvals = hessian_matrix_eigvals(H_elems)
        return eigvals  # list of 3 arrays

    eigs_s2 = get_eigenvalues(img, sigma=2.0)  # 3
    features.extend(eigs_s2)

    eigs_s4 = get_eigenvalues(img, sigma=4.0)  # 3
    features.extend(eigs_s4)

    stack = np.stack(features, axis=-1).astype(np.float32)

    # STRICT check
    if stack.shape[-1] != 40:
        raise RuntimeError(f"Features count is {stack.shape[-1]}, expected 40.")

    return stack


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


# ============================================================
# NEW: Save fold visualization (validation crop only)
# ============================================================


def save_fold_visualizations(
    clf,
    fold_idx: int,
    val_crop_id: str,
    val_crop_data: dict,
    n_feats: int,
    cmap,
    legend_patches,
    num_vis: int = 10,
):
    """
    Save visualizations for the validation crop of one fold only.
    Output:
      RUN_DIR/fold_XX/<crop_id>/*.png
    """
    # 直接在 RUN_DIR 下按 crop 建目录（不再区分 fold）
    crop_dir = os.path.join(RUN_DIR, val_crop_id)
    os.makedirs(crop_dir, exist_ok=True)


    raw_crop_vis = val_crop_data["raw"]
    label_multi_vis = val_crop_data["label"]
    Dz, Dy, Dx = val_crop_data["shape"]
    feat_vol_4d = val_crop_data["features_4d"]

    # Uniform sampling along Z-axis:
    # Select num_vis slices evenly from top (z=0) to bottom (z=Dz-1), sorted in ascending order.
    if Dz <= num_vis:
        vis_zs = list(range(Dz))
    else:
        vis_zs = np.linspace(0, Dz - 1, num_vis)
        vis_zs = np.round(vis_zs).astype(int)

        # Remove duplicates caused by rounding and keep unique indices
        vis_zs = np.unique(vis_zs)

        # If the number of unique slices is smaller than num_vis (rare case),
        # fill the missing ones by selecting nearest unused indices.
        if len(vis_zs) < num_vis:
            candidates = [z for z in range(Dz) if z not in set(vis_zs)]

            # Target positions for approximately uniform distribution
            target = np.linspace(0, Dz - 1, num_vis)
            target = np.round(target).astype(int)

            for t in target:
                if len(vis_zs) >= num_vis or not candidates:
                    break
                # Choose the candidate index closest to the target position
                nearest = min(candidates, key=lambda x: abs(x - t))
                vis_zs = np.append(vis_zs, nearest)
                candidates.remove(nearest)

            vis_zs = np.unique(vis_zs)

        # Final safety: ensure exactly num_vis slices in sorted order
        vis_zs = np.sort(vis_zs)[:num_vis].tolist()



    for i, vis_z in enumerate(vis_zs):
        feat_vis_slice = feat_vol_4d[vis_z, :, :, :]
        X_vis = feat_vis_slice.reshape(-1, n_feats)

        y_pred_vis = clf.predict(X_vis).reshape(Dy, Dx)
        y_pred_vis = median_filter(y_pred_vis, size=3)

        y_gt_vis = label_multi_vis[vis_z]
        raw_vis = raw_crop_vis[vis_z]

        fig = plt.figure(figsize=(20, 6))
        plt.suptitle(f"RF | Fold {fold_idx:02d} | Crop {val_crop_id} | z={vis_z}", fontsize=16)

        plt.subplot(1, 3, 1)
        plt.title("Raw")
        plt.imshow(raw_vis, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Ground Truth")
        plt.imshow(y_gt_vis, cmap=cmap, vmin=0, vmax=9)
        plt.axis("off")
        plt.legend(
            handles=legend_patches,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            title="Classes (ID)",
            borderaxespad=0.0,
        )

        plt.subplot(1, 3, 3)
        plt.title("Prediction")
        plt.imshow(y_pred_vis, cmap=cmap, vmin=0, vmax=9)
        plt.axis("off")
        plt.legend(
            handles=legend_patches,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            title="Classes (ID)",
            borderaxespad=0.0,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.94])

        save_path = os.path.join(crop_dir, f"slice_{i+1:02d}_z{vis_z:04d}.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"    Saved: {save_path}")


# ============================================================
# Step 2-4: Load multiple crops, align, and store (MODIFIED for s1)
# ============================================================

all_crops_data = []
crop_pixel_counts = []

SCALE_FACTOR = 2  # s1 is 1/2 of s0

print("\n===== Loading and Aligning Multiple Crops (s1 Resolution) =====")

for crop_id in CROP_IDS:
    print(f"\nProcessing {crop_id}...")
    CROP_ROOT = os.path.join(GROUNDTRUTH_ROOT, crop_id)
    REF_S0 = os.path.join(CROP_ROOT, REF_CLASS, "s0")
    REF_ZATTR = os.path.join(CROP_ROOT, REF_CLASS, ".zattrs")

    # 1. Open reference zarr to get s0 shape
    ref_zarr_s0 = zarr.open(REF_S0, mode="r")
    Dz_s0, Dy_s0, Dx_s0 = ref_zarr_s0.shape

    # 2. Read .zattrs (translation + scale based on s0)
    with open(REF_ZATTR, "r") as f:
        attrs = json.load(f)
    ms = attrs["multiscales"][0]["datasets"][0]
    scale = ms["coordinateTransformations"][0]["scale"]
    trans = ms["coordinateTransformations"][1]["translation"]
    scale_z, scale_y, scale_x = scale
    tz, ty, tx = trans

    # 3. Convert nm -> s0 voxel index -> s1 voxel index
    vz0_s0 = int(tz / scale_z)
    vy0_s0 = int(ty / scale_y)
    vx0_s0 = int(tx / scale_x)

    vz0 = vz0_s0 // SCALE_FACTOR
    vy0 = vy0_s0 // SCALE_FACTOR
    vx0 = vx0_s0 // SCALE_FACTOR

    Dz = Dz_s0 // SCALE_FACTOR
    Dy = Dy_s0 // SCALE_FACTOR
    Dx = Dx_s0 // SCALE_FACTOR

    vz1, vy1, vx1 = vz0 + Dz, vy0 + Dy, vx0 + Dx

    raw_crop = raw_zarr[vz0:vz1, vy0:vy1, vx0:vx1]
    print(f"  Raw crop shape (s1): {raw_crop.shape}")

    # 4. Build multi-class label (s0 -> downsample -> s1)
    label_multi = np.zeros((Dz, Dy, Dx), dtype=np.uint8)
    for cname, real_id in SELECT_CLASSES.items():
        path = os.path.join(CROP_ROOT, cname, "s0")
        try:
            arr_s0 = zarr.open(path, mode="r")[:]
            arr_s1 = arr_s0[::SCALE_FACTOR, ::SCALE_FACTOR, ::SCALE_FACTOR]
            arr_s1 = arr_s1[:Dz, :Dy, :Dx]
            cid = CLASS_ID_MAP[cname]
            label_multi[arr_s1 > 0] = cid
        except Exception as e:
            print(f"  Warning: Failed to load class {cname} in {crop_id}. Error: {e}")

    # Shape safety
    if raw_crop.shape != label_multi.shape:
        print(f"  ! Shape Mismatch Fixed: Raw {raw_crop.shape} vs Label {label_multi.shape}")
        min_z = min(raw_crop.shape[0], label_multi.shape[0])
        min_y = min(raw_crop.shape[1], label_multi.shape[1])
        min_x = min(raw_crop.shape[2], label_multi.shape[2])
        raw_crop = raw_crop[:min_z, :min_y, :min_x]
        label_multi = label_multi[:min_z, :min_y, :min_x]
        Dz, Dy, Dx = raw_crop.shape

    print(f"  Unique final multi-class labels in {crop_id}: {np.unique(label_multi)}")

    all_crops_data.append({"raw": raw_crop, "label": label_multi, "shape": (Dz, Dy, Dx), "id": crop_id})
    crop_pixel_counts.append(Dz * Dy * Dx)

# ============================================================
# Step 5: Full 3D Feature Extraction and Merging
# ============================================================
print("\n===== Performing Feature Extraction and Merging =====")

X_all_crops_list = []
y_all_crops_list = []
n_feats = 0

for crop_item in all_crops_data:
    raw_crop = crop_item["raw"]
    label_multi = crop_item["label"]
    print(f"  Extracting features for {crop_item['id']}...")

    X_vol_4d = extract_full_3d_features_40ch(raw_crop)
    crop_item["features_4d"] = X_vol_4d

    if n_feats == 0:
        n_feats = X_vol_4d.shape[-1]
        print(f"Total features per voxel: {n_feats}")

    X_pixels = X_vol_4d.reshape(-1, n_feats)
    y_pixels = label_multi.reshape(-1)
    X_all_crops_list.append(X_pixels)
    y_all_crops_list.append(y_pixels)

X_all_flat = np.concatenate(X_all_crops_list, axis=0)
y_all_flat = np.concatenate(y_all_crops_list, axis=0)
print(f"Total merged flat dataset size: {X_all_flat.shape[0]} pixels")

# ============================================================
# Step 6: K-fold Cross Validation Setup (Crop-based)
# ============================================================
crop_indices = np.arange(len(CROP_IDS))
kf = KFold(n_splits=len(CROP_IDS), shuffle=True, random_state=42)

crop_start_indices = np.cumsum([0] + crop_pixel_counts)
fold_scores = []

# ============================================================
# Step 7: Train & Validate (Crop-based split & Chunking)
# ============================================================

for fold, (train_crop_idx, val_crop_idx) in tqdm(
    enumerate(kf.split(crop_indices)), total=kf.n_splits, desc="K-Fold Progress"
):
    print(f"\n================ FOLD {fold:02d} ================")
    fold_start = time.time()

    # 1) Build training set from training crops
    X_train_list, y_train_list = [], []
    for c_idx in train_crop_idx:
        start = crop_start_indices[c_idx]
        end = crop_start_indices[c_idx + 1]
        X_train_list.append(X_all_flat[start:end])
        y_train_list.append(y_all_flat[start:end])
    X_train_full = np.concatenate(X_train_list, axis=0)
    y_train_full = np.concatenate(y_train_list, axis=0)

    # 2) Build validation set from the single val crop
    val_c_idx = val_crop_idx[0]
    start = crop_start_indices[val_c_idx]
    end = crop_start_indices[val_c_idx + 1]
    X_val = X_all_flat[start:end]
    y_val = y_all_flat[start:end]
    val_crop_id = CROP_IDS[val_c_idx]
    print(f"FOLD {fold:02d}: Validation Crop ID: {val_crop_id}")

    # A) Sampling / limit
    PIXEL_LIMIT = 50000
    print(f"FOLD {fold:02d}: Applying pixel limit of {PIXEL_LIMIT} per class to training data...")
    X_train, y_train = limit_class_pixels(
        X_train_full, y_train_full, max_per_class=PIXEL_LIMIT, random_state=42 + fold
    )
    print(f"Fold {fold:02d} Sampled Train set shape:", X_train.shape)
    print(f"Fold {fold:02d} Val set shape:  ", X_val.shape)

    # C) Train RandomForest
    print("Training RandomForest...")
    clf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", n_jobs=2, random_state=42
    )
    clf.fit(X_train, y_train)

    # D/E) Predict in chunks
    print("Predicting...")
    chunk_size = 8000000
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

    # Reshape back to 3D for post-processing
    val_crop_data = all_crops_data[val_c_idx]
    Dz_val, Dy_val, Dx_val = val_crop_data["shape"]
    y_pred_3d = y_pred_flat.reshape(Dz_val, Dy_val, Dx_val)

    print("Applying Post-processing (Median Filter)...")
    y_pred_3d_refined = median_filter(y_pred_3d, size=3)
    y_pred_final = y_pred_3d_refined.reshape(-1)

    print(f"Prediction complete. Final shape: {y_pred_final.shape}")

    # Metrics
    iou_per_class = jaccard_score(
        y_val, y_pred_final, average=None, labels=labels_eval, zero_division=0
    )
    dice_per_class = f1_score(
        y_val, y_pred_final, average=None, labels=labels_eval, zero_division=0
    )
    iou_macro = float(np.mean(iou_per_class))
    dice_macro = float(np.mean(dice_per_class))
    print(f"FOLD {fold:02d} Result: IoU_macro={iou_macro:.4f}, Dice_macro={dice_macro:.4f}")

    fold_scores.append(
        {
            "iou_macro": iou_macro,
            "dice_macro": dice_macro,
            "iou_per_class": iou_per_class,
            "dice_per_class": dice_per_class,
        }
    )

    # ✅ Save visualization for THIS fold's validation crop only
    print(f"Saving fold-{fold:02d} visualization for {val_crop_id} ...")
    save_fold_visualizations(
        clf=clf,
        fold_idx=fold,
        val_crop_id=val_crop_id,
        val_crop_data=val_crop_data,
        n_feats=n_feats,
        cmap=cmap,
        legend_patches=legend_patches,
        num_vis=10,
    )

    elapsed = time.time() - fold_start
    print(f"⏱ Fold {fold:02d} took {elapsed / 60:.2f} minutes")

# ============================================================
# Step 7.5: Train final model on ALL data (Balanced Sampling)
# ============================================================
print("\n===== Step 7.5: Training final model on ALL data (Balanced Sampling) =====")

X_sub, y_sub = limit_class_pixels(X_all_flat, y_all_flat, max_per_class=50000)

print("Training FINAL RandomForest model...")
final_clf = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    n_jobs=2,
    random_state=42,
)
final_clf.fit(X_sub, y_sub)
print("🎉 Final model training complete!")

# ============================================================
# Step 8: Print final aggregated results
# ============================================================
print("\n================ Final K-fold Results ================\n")

all_iou_per_class = np.array([s["iou_per_class"] for s in fold_scores])
all_dice_per_class = np.array([s["dice_per_class"] for s in fold_scores])

avg_iou_per_class = np.mean(all_iou_per_class, axis=0)
avg_dice_per_class = np.mean(all_dice_per_class, axis=0)

for k, scores in enumerate(fold_scores):
    print(f"Fold {k:02d}:   IoU_macro={scores['iou_macro']:.4f}   Dice_macro={scores['dice_macro']:.4f}")

print("\n---------------- Detailed Class Averages ----------------")
print("Class | Avg IoU (Jaccard) | Avg Dice (F1)")
print("------|-------------------|---------------")
for i, cname in enumerate(CLASS_NAMES_ORDERED):
    print(f"{cname:8s}| {avg_iou_per_class[i]:.4f} | {avg_dice_per_class[i]:.4f}")

avg_macro_iou = float(np.mean(avg_iou_per_class))
avg_macro_dice = float(np.mean(avg_dice_per_class))

print("\n================ Aggregated Macro Scores ================")
print(f"Average Macro IoU (5 classes): {avg_macro_iou:.4f}")
print(f"Average Macro Dice (5 classes): {avg_macro_dice:.4f}")

print(f"\n✅ All fold images saved under:\n{RUN_DIR}\n")
print("\n--- Script Finished ---")
