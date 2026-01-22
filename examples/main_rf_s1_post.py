# =============================================
#   CellMap nucleus-related segmentation (Multi-Crop K-Fold)
#   Target classes (real IDs): cyto(35), mito_mem(3), mito_lum(4), er_mem(16), er_lum(17)
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
from datetime import datetime
from scipy.ndimage import gaussian_filter, sobel, laplace  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.model_selection import KFold  # type: ignore
from sklearn.metrics import jaccard_score, f1_score  # type: ignore
from scipy.ndimage import (
    maximum_filter,
    minimum_filter,
    binary_fill_holes,
    distance_transform_edt,
)  # type: ignore
from skimage.feature import (
    hessian_matrix,
    hessian_matrix_eigvals,
)  # type: ignore
from skimage.morphology import (
    binary_opening,
    binary_closing,
    binary_dilation,
    disk,
    ball,
    remove_small_holes,
    remove_small_objects,
)  # type: ignore

# ---------------------------------------------
# Step 1. Setup paths
# ---------------------------------------------

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

RAW_S1 = r"../data/jrc_cos7-1a/jrc_cos7-1a.zarr/recon-1/em/fibsem-uint8/s1"
GROUNDTRUTH_ROOT = r"../data/jrc_cos7-1a/jrc_cos7-1a.zarr/recon-1/labels/groundtruth"

# Real IDs in dataset (for reading binary masks)
SELECT_CLASSES = {
    "cyto": 35,
    "mito_mem": 3,
    "mito_lum": 4,
    "er_mem": 16,
    "er_lum": 17,
}

# Internal class IDs used in training/eval labels
CLASS_ID_MAP = {
    "cyto": 1,
    "mito_mem": 2,
    "mito_lum": 3,
    "er_mem": 4,
    "er_lum": 5,
}

CLASS_NAMES_ORDERED = ["cyto", "mito_mem", "mito_lum", "er_mem", "er_lum"]
labels_eval = [1, 2, 3, 4, 5]

# Use a class that definitely exists as reference for crop bounds/zattrs
REF_CLASS = "cyto"

raw_zarr = zarr.open(RAW_S1, mode="r")
print("Raw shape:", raw_zarr.shape)

SCALE_FACTOR = 2  # s1 is downsampled from s0 by factor 2

# ============================================================
# Feature extraction (40 channels)
# ============================================================

def extract_full_3d_features_40ch(vol_3d: np.ndarray) -> np.ndarray:
    """
    Feature set: 40 channels
    - Raw + Gaussians
    - Gradient magnitude
    - Laplacian of Gaussian
    - DoG
    - Hessian diagonal approx
    - Local std
    - Local range texture
    - Hessian eigenvalues (2 scales)
    """
    img = vol_3d.astype(np.float32) / 255.0
    features = []

    sigmas = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0]
    gaussians = {}

    # Group 1: Raw + Gaussians (1 + 8 = 9 ch)
    features.append(img)
    for s in sigmas:
        g = gaussian_filter(img, sigma=s)
        gaussians[s] = g
        features.append(g)

    # Group 2: Gradient magnitude (5 ch)
    def get_grad_mag(smoothed_vol: np.ndarray) -> np.ndarray:
        gz = sobel(smoothed_vol, axis=0)
        gy = sobel(smoothed_vol, axis=1)
        gx = sobel(smoothed_vol, axis=2)
        return np.sqrt(gz**2 + gy**2 + gx**2)

    for s in [1.0, 2.0, 4.0, 8.0, 12.0]:
        features.append(get_grad_mag(gaussians[s]))

    # Group 3: LoG (5 ch)
    for s in [1.0, 2.0, 4.0, 8.0, 12.0]:
        features.append(laplace(gaussians[s]))

    # Group 4: DoG (5 ch)
    dog_pairs = [(1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 16.0)]
    for s1, s2 in dog_pairs:
        features.append(gaussians[s1] - gaussians[s2])

    # Group 5: Hessian diagonal approx (6 ch)
    def get_hessian_diag(smoothed_vol: np.ndarray):
        zz = sobel(sobel(smoothed_vol, axis=0), axis=0)
        yy = sobel(sobel(smoothed_vol, axis=1), axis=1)
        xx = sobel(sobel(smoothed_vol, axis=2), axis=2)
        return xx, yy, zz

    hxx, hyy, hzz = get_hessian_diag(gaussians[2.0])
    features.extend([hxx, hyy, hzz])
    hxx, hyy, hzz = get_hessian_diag(gaussians[4.0])
    features.extend([hxx, hyy, hzz])

    # Group 6: Local std (2 ch)
    def get_std(raw: np.ndarray, smoothed: np.ndarray, s: float) -> np.ndarray:
        mean_sq = gaussian_filter(raw**2, sigma=s)
        mean = smoothed
        var = mean_sq - mean**2
        return np.sqrt(np.maximum(var, 0))

    features.append(get_std(img, gaussians[2.0], 2.0))
    features.append(get_std(img, gaussians[4.0], 4.0))

    # Group 7: Local range texture (2 ch)
    def get_range(vol: np.ndarray, size: int) -> np.ndarray:
        mx = maximum_filter(vol, size=size)
        mn = minimum_filter(vol, size=size)
        return mx - mn

    features.append(get_range(img, size=3))
    features.append(get_range(gaussians[1.0], size=5))

    # Group 8: Hessian eigenvalues (6 ch)
    def get_eigenvalues(vol: np.ndarray, sigma: float):
        H_elems = hessian_matrix(vol, sigma=sigma, order="rc", use_gaussian_derivatives=False)
        eigvals = hessian_matrix_eigvals(H_elems)
        return eigvals

    features.extend(get_eigenvalues(img, sigma=2.0))
    features.extend(get_eigenvalues(img, sigma=4.0))

    stack = np.stack(features, axis=-1).astype(np.float32)
    if stack.shape[-1] != 40:
        print(f"Warning: Features count is {stack.shape[-1]}, expected 40.")
        stack = stack[..., :40]

    return stack

# ============================================================
# Post-processing: full pipeline (recommended)
# ============================================================

def classwise_refine_and_compose(
    y_3d: np.ndarray,
    priority: list[int],
    radius: int = 1,
    min_size_map: dict[int, int] | None = None,
    do_3d: bool = False,
) -> np.ndarray:
    """
    Apply binary opening/closing + remove_small_objects per class,
    then compose labels by priority (higher priority wins).
    """
    y = y_3d.copy()
    out = np.zeros_like(y)

    if min_size_map is None:
        min_size_map = {}

    if do_3d:
        se = ball(radius)
        for cid in priority:
            mask = (y == cid)
            mask = binary_opening(mask, se)
            mask = binary_closing(mask, se)
            ms = int(min_size_map.get(cid, 0))
            if ms > 0:
                mask = remove_small_objects(mask, min_size=ms)
            out[(out == 0) & mask] = cid
        return out

    se2d = disk(radius)
    for z in range(y.shape[0]):
        slice_out = np.zeros_like(y[z])
        for cid in priority:
            mask = (y[z] == cid)
            mask = binary_opening(mask, se2d)
            mask = binary_closing(mask, se2d)
            ms = int(min_size_map.get(cid, 0))
            if ms > 0:
                mask = remove_small_objects(mask, min_size=ms)
            slice_out[(slice_out == 0) & mask] = cid
        out[z] = slice_out

    return out


def enforce_mem_lumen_constraint_v2(
    y_3d: np.ndarray,
    mem_id: int,
    lum_id: int,
    protect_ids=(),
    allowed_overwrite_ids=(0, 1),
    do_3d: bool = False,
    closing_radius: int = 2,
    min_mem_size: int = 64,
    min_lumen_area: int = 128,
    max_lumen_dist_to_mem: int | None = 30,
) -> np.ndarray:
    """
    Enforce: enclosed regions by membrane (mem_id) become lumen (lum_id).
    Enhancements:
    - remove small membrane fragments before closing
    - optional distance constraint: lumen must be within max distance to membrane
    - only overwrite allowed classes and never overwrite protected classes
    """
    y = y_3d.copy()

    if do_3d:
        mem = (y == mem_id)
        mem = remove_small_objects(mem, min_size=min_mem_size)

        mem_closed = binary_closing(mem, ball(closing_radius))
        filled = binary_fill_holes(mem_closed)
        lumen = filled & (~mem_closed)

        if min_lumen_area > 0:
            lumen = remove_small_holes(lumen, area_threshold=min_lumen_area)

        if max_lumen_dist_to_mem is not None:
            dist = distance_transform_edt(~mem_closed)
            lumen = lumen & (dist <= max_lumen_dist_to_mem)

        protect = np.zeros_like(y, dtype=bool)
        for pid in protect_ids:
            protect |= (y == pid)

        can_overwrite = np.isin(y, allowed_overwrite_ids)
        write_mask = lumen & can_overwrite & (~protect) & (y != mem_id)
        y[write_mask] = lum_id
        return y

    se2d = disk(closing_radius)
    for z in range(y.shape[0]):
        mem = (y[z] == mem_id)
        mem = remove_small_objects(mem, min_size=min_mem_size)

        mem_closed = binary_closing(mem, se2d)
        filled = binary_fill_holes(mem_closed)
        lumen = filled & (~mem_closed)

        if min_lumen_area > 0:
            lumen = remove_small_holes(lumen, area_threshold=min_lumen_area)

        if max_lumen_dist_to_mem is not None:
            dist = distance_transform_edt(~mem_closed)
            lumen = lumen & (dist <= max_lumen_dist_to_mem)

        protect = np.zeros_like(y[z], dtype=bool)
        for pid in protect_ids:
            protect |= (y[z] == pid)

        can_overwrite = np.isin(y[z], allowed_overwrite_ids)
        write_mask = lumen & can_overwrite & (~protect) & (y[z] != mem_id)
        y[z][write_mask] = lum_id

    return y


def enforce_lumen_near_membrane(
    y_3d: np.ndarray,
    mem_id: int,
    lum_id: int,
    fallback_id: int = 1,  # cyto
    near_radius: int = 2,
    do_3d: bool = False,
) -> np.ndarray:
    """
    Any lumen pixels not near membrane are considered false positives and reset to fallback_id.
    """
    y = y_3d.copy()

    if do_3d:
        mem = (y == mem_id)
        mem_near = binary_dilation(mem, ball(near_radius))
        fake_lumen = (y == lum_id) & (~mem_near)
        y[fake_lumen] = fallback_id
        return y

    se2d = disk(near_radius)
    for z in range(y.shape[0]):
        mem = (y[z] == mem_id)
        mem_near = binary_dilation(mem, se2d)
        fake_lumen = (y[z] == lum_id) & (~mem_near)
        y[z][fake_lumen] = fallback_id

    return y


def full_postprocess_pipeline(
    y_3d: np.ndarray,
    do_3d: bool = False,
) -> np.ndarray:
    """
    Full post-processing pipeline:
    1) classwise refine + priority compose
    2) mito membrane -> lumen fill
    3) ER membrane -> lumen fill
    4) lumen must be near corresponding membrane (otherwise reset to cyto)
    5) final small-object cleanup (slice-wise)
    """
    y = y_3d.copy()

    # (1) Class-wise refine + priority compose
    priority = [4, 2, 5, 3, 1]  # er_mem, mito_mem, er_lum, mito_lum, cyto
    min_size_map = {
        4: 128,  # er_mem
        2: 128,  # mito_mem
        5: 128,  # er_lum
        3: 128,  # mito_lum
        1: 256,  # cyto
    }
    y = classwise_refine_and_compose(
        y,
        priority=priority,
        radius=1,
        min_size_map=min_size_map,
        do_3d=do_3d,
    )

    # (2) Mito mem -> mito lumen
    y = enforce_mem_lumen_constraint_v2(
        y,
        mem_id=2,  # mito_mem
        lum_id=3,  # mito_lum
        protect_ids=(4, 5),  # protect ER
        allowed_overwrite_ids=(0, 1, 2, 3),
        do_3d=do_3d,
        closing_radius=1,
        min_mem_size=64,
        min_lumen_area=64,
        max_lumen_dist_to_mem=20,
    )

    # (3) ER mem -> ER lumen
    y = enforce_mem_lumen_constraint_v2(
        y,
        mem_id=4,  # er_mem
        lum_id=5,  # er_lum
        protect_ids=(2, 3),  # protect mito
        allowed_overwrite_ids=(0, 1, 4, 5),
        do_3d=do_3d,
        closing_radius=2,
        min_mem_size=96,
        min_lumen_area=128,
        max_lumen_dist_to_mem=30,
    )

    # (4) Lumen must be near membrane
    y = enforce_lumen_near_membrane(y, mem_id=2, lum_id=3, fallback_id=1, near_radius=2, do_3d=do_3d)
    y = enforce_lumen_near_membrane(y, mem_id=4, lum_id=5, fallback_id=1, near_radius=2, do_3d=do_3d)

    # (5) Final cleanup (slice-wise)
    if not do_3d:
        for z in range(y.shape[0]):
            for cid, ms in [(2, 128), (3, 128), (4, 128), (5, 128)]:
                mask = (y[z] == cid)
                cleaned = remove_small_objects(mask, min_size=ms)
                # Reset removed pixels to cyto
                y[z][mask & (~cleaned)] = 1

    return y

# ============================================================
# Step 2-4: Load multiple crops, align, and store (s1)
# ============================================================

all_crops_data = []
crop_pixel_counts = []

print("\n===== Loading and Aligning Multiple Crops (s1 Resolution) =====")

for crop_id in CROP_IDS:
    print(f"\nProcessing {crop_id}...")
    CROP_ROOT = os.path.join(GROUNDTRUTH_ROOT, crop_id)
    REF_S0 = os.path.join(CROP_ROOT, REF_CLASS, "s0")
    REF_ZATTR = os.path.join(CROP_ROOT, REF_CLASS, ".zattrs")

    ref_zarr_s0 = zarr.open(REF_S0, mode="r")
    Dz_s0, Dy_s0, Dx_s0 = ref_zarr_s0.shape

    with open(REF_ZATTR, "r") as f:
        attrs = json.load(f)
    ms = attrs["multiscales"][0]["datasets"][0]
    scale = ms["coordinateTransformations"][0]["scale"]
    trans = ms["coordinateTransformations"][1]["translation"]
    scale_z, scale_y, scale_x = scale
    tz, ty, tx = trans

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

    label_multi = np.zeros((Dz, Dy, Dx), dtype=np.uint8)
    for cname, _real_id in SELECT_CLASSES.items():
        path = os.path.join(CROP_ROOT, cname, "s0")
        try:
            arr_s0 = zarr.open(path, mode="r")[:]
            arr_s1 = arr_s0[::SCALE_FACTOR, ::SCALE_FACTOR, ::SCALE_FACTOR]
            arr_s1 = arr_s1[:Dz, :Dy, :Dx]
            cid = CLASS_ID_MAP[cname]
            label_multi[arr_s1 > 0] = cid
        except Exception as e:
            print(f"  Warning: Failed to load class {cname} in {crop_id}. Error: {e}")

    if raw_crop.shape != label_multi.shape:
        print(f"  ! Shape mismatch fixed: Raw {raw_crop.shape} vs Label {label_multi.shape}")
        min_z = min(raw_crop.shape[0], label_multi.shape[0])
        min_y = min(raw_crop.shape[1], label_multi.shape[1])
        min_x = min(raw_crop.shape[2], label_multi.shape[2])
        raw_crop = raw_crop[:min_z, :min_y, :min_x]
        label_multi = label_multi[:min_z, :min_y, :min_x]
        Dz, Dy, Dx = raw_crop.shape

    print(f"  Unique final labels in {crop_id}: {np.unique(label_multi)}")

    all_crops_data.append(
        {"raw": raw_crop, "label": label_multi, "shape": (Dz, Dy, Dx), "id": crop_id}
    )
    crop_pixel_counts.append(Dz * Dy * Dx)

# ============================================================
# Step 5: Determine feature count
# ============================================================

print("\n===== Step 5: Determining Feature Count =====")
tmp_raw = all_crops_data[0]["raw"]
tmp_feat = extract_full_3d_features_40ch(tmp_raw)
n_feats = tmp_feat.shape[-1]
del tmp_feat
print(f"Total features per voxel: {n_feats}")

# ============================================================
# Sampling
# ============================================================

def sample_from_crop(X_flat: np.ndarray, y_flat: np.ndarray, max_per_class: int = 50000, random_state: int = 42):
    """
    Sample up to max_per_class pixels per class from one crop.
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
# Step 6: K-fold setup
# ============================================================

crop_indices = np.arange(len(CROP_IDS))
kf = KFold(n_splits=len(CROP_IDS), shuffle=True, random_state=42)
fold_scores = []

# ============================================================
# Step 7: Train & Validate
# ============================================================

for fold, (train_crop_idx, val_crop_idx) in tqdm(
    enumerate(kf.split(crop_indices)), total=kf.n_splits, desc="K-Fold Progress"
):
    print(f"\n================ FOLD {fold} ================")
    fold_start = time.time()

    val_c_idx = val_crop_idx[0]
    print(f"FOLD {fold}: Validation Crop ID: {CROP_IDS[val_c_idx]}")

    PIXEL_LIMIT = 50000
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

        X_s, y_s = sample_from_crop(
            X_flat, y_flat, max_per_class=PIXEL_LIMIT, random_state=42 + fold + c_idx
        )

        X_train_parts.append(X_s)
        y_train_parts.append(y_s)

        del X_vol_4d, X_flat, y_flat, X_s, y_s

    X_train = np.concatenate(X_train_parts, axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)
    del X_train_parts, y_train_parts

    print(f"Fold {fold} Sampled Train set shape: {X_train.shape}")

    val_crop_data = all_crops_data[val_c_idx]
    Dz_val, Dy_val, Dx_val = val_crop_data["shape"]

    print(f"  [Val] Extracting features for {val_crop_data['id']}...")
    X_val_4d = extract_full_3d_features_40ch(val_crop_data["raw"])
    X_val = X_val_4d.reshape(-1, n_feats)
    y_val = val_crop_data["label"].reshape(-1)
    del X_val_4d

    print(f"Fold {fold} Val set shape: {X_val.shape}")

    print("Training RandomForest...")
    clf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        n_jobs=2,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    del X_train, y_train

    print("Predicting...")
    chunk_size = 2_000_000
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
    del y_pred_list, X_val

    y_pred_3d = y_pred_flat.reshape(Dz_val, Dy_val, Dx_val)
    del y_pred_flat

    print("Applying Post-processing (Full Pipeline)...")
    y_pred_3d_refined = full_postprocess_pipeline(y_pred_3d, do_3d=False)
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
    iou_macro = float(np.mean(iou_per_class))
    dice_macro = float(np.mean(dice_per_class))

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
    print(f"Fold {fold} took {elapsed / 60:.2f} minutes")

    del y_val, y_pred_final, clf

# ============================================================
# Step 7.5: Train final model on all crops
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

print("Final model training complete!")

# ============================================================
# Step 8: Print aggregated results
# ============================================================

print("\n================ Final K-fold Results ================\n")

all_iou_per_class = np.array([s["iou_per_class"] for s in fold_scores])
all_dice_per_class = np.array([s["dice_per_class"] for s in fold_scores])

avg_iou_per_class = np.mean(all_iou_per_class, axis=0)
avg_dice_per_class = np.mean(all_dice_per_class, axis=0)

for k, scores in enumerate(fold_scores):
    print(f"Fold {k}: IoU_macro={scores['iou_macro']:.4f} Dice_macro={scores['dice_macro']:.4f}")

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

# ============================================================
# Step 9: Save prediction slices (Run-based folder structure)
# Output: ../Result/RF/run_YYYYMMDD_HHMMSS/crop*/slice_*.png
# ============================================================

result_root = "../Result/RF"
os.makedirs(result_root, exist_ok=True)

run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_folder = os.path.join(result_root, f"run_{run_time}")
os.makedirs(experiment_folder, exist_ok=True)

print(f"\nSaving visualizations into: {experiment_folder}")

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
        {
            "model_idx": idx,
            "real_id": real_id,
            "label": f"{name} ({real_id})",
            "color": cmap(idx),
        }
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

num_vis_per_crop = 10
crops_data_dict = {item["id"]: item for item in all_crops_data}

for crop_id in CROP_IDS:
    crop_item = crops_data_dict[crop_id]
    crop_save_folder = os.path.join(experiment_folder, crop_id)
    os.makedirs(crop_save_folder, exist_ok=True)
    print(f"\n--- Processing {crop_id} (Saving {num_vis_per_crop} slices) ---")

    raw_crop_vis = crop_item["raw"]
    label_multi_vis = crop_item["label"]
    Dz, Dy, Dx = crop_item["shape"]

    feat_vol_4d = extract_full_3d_features_40ch(raw_crop_vis)

    # Choose exactly num_vis_per_crop unique z indices
    if Dz <= num_vis_per_crop:
        vis_zs = list(range(Dz))
    else:
        vis_zs = np.linspace(0, Dz - 1, num=num_vis_per_crop, dtype=int).tolist()

    # Stable unique
    vis_zs = sorted(list(dict.fromkeys(vis_zs)))
    if len(vis_zs) < num_vis_per_crop:
        rng = np.random.RandomState(123)
        candidates = [z for z in range(Dz) if z not in vis_zs]
        extra = rng.choice(candidates, size=(num_vis_per_crop - len(vis_zs)), replace=False)
        vis_zs.extend(extra.tolist())
        vis_zs = sorted(vis_zs)
    elif len(vis_zs) > num_vis_per_crop:
        vis_zs = vis_zs[:num_vis_per_crop]

    for i, vis_z in enumerate(vis_zs):
        feat_vis_slice = feat_vol_4d[vis_z, :, :, :]
        X_vis = feat_vis_slice.reshape(-1, n_feats)

        y_pred_vis = final_clf.predict(X_vis).reshape(Dy, Dx)

        # Apply the same full pipeline on the single slice (wrap 2D->3D->unwrap)
        y_pred_vis = full_postprocess_pipeline(y_pred_vis[np.newaxis, ...], do_3d=False)[0]

        y_gt_vis = label_multi_vis[vis_z]
        raw_vis = raw_crop_vis[vis_z]

        fig = plt.figure(figsize=(20, 6))
        plt.suptitle(f"Run: run_{run_time} | Crop: {crop_id} | z={vis_z}", fontsize=16)

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

        save_path = os.path.join(crop_save_folder, f"slice_{i + 1:02d}_z{vis_z:04d}.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"  [{i + 1}/{len(vis_zs)}] Saved: {save_path}")

    # Free features to avoid memory accumulation
    del feat_vol_4d

print("\n--- Script Finished ---")
