import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- Configuration ---
MODEL_PATH = "/home/chwa386g/chwa386g/cellmap-segmentation-challenge/Result/unet3d_runs/run_20260111_122259/best.pt"
OUTPUT_DIR = "./predictions_output"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Adjust these based on your training settings
CLASS_NAMES = ["Background", "Class1", "Class2", "Class3", "Class4", "Class5"] 
N_VAL_SLICES = 5  # Number of slices to visualize

def load_model(path, device):
    """Load the trained model."""
    # If you saved the entire model using torch.jit.save
    model = torch.jit.load(path, map_location=device)
    # If you saved state_dict, you'll need to instantiate the model class first:
    # model = UNet3D(...) 
    # model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def predict_volume(model, volume_tensor):
    """
    Perform inference on a 3D volume.
    volume_tensor: [C, D, H, W]
    """
    with torch.no_grad():
        # Add batch dimension: [1, C, D, H, W]
        input_batch = volume_tensor.unsqueeze(0).to(DEVICE)
        output = model(input_batch)
        
        # Assuming output is logits, get the class indices
        # If output is [1, NumClasses, D, H, W]
        pred = torch.argmax(output, dim=1).squeeze(0) 
    return pred.cpu().numpy()

def save_visualizations(raw, label, pred, run_dir, crop_id="inference"):
    """
    Visualize slices and save to disk (based on your snippet).
    """
    vis_dir = os.path.join(run_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    Dz = raw.shape[0]
    zs = np.linspace(0, Dz - 1, N_VAL_SLICES, dtype=int)

    cmap = plt.get_cmap("tab10")
    legend_patches = []
    for i, name in enumerate(CLASS_NAMES):
        legend_patches.append(mpatches.Patch(color=cmap(i), label=f"{name} ({i})"))

    for z in zs:
        fig = plt.figure(figsize=(20, 6))

        # --- Raw ---
        plt.subplot(1, 3, 1)
        plt.title(f"Raw (Slice {z})")
        plt.imshow(raw[z], cmap="gray")
        plt.axis("off")

        # --- GT (Ground Truth - if available) ---
        plt.subplot(1, 3, 2)
        plt.title("GT (Optional)")
        if label is not None:
            plt.imshow(label[z], cmap=cmap, vmin=0, vmax=9)
        plt.axis("off")
        plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', title="Classes")

        # --- Pred ---
        plt.subplot(1, 3, 3)
        plt.title("Pred")
        plt.imshow(pred[z], cmap=cmap, vmin=0, vmax=9)
        plt.axis("off")
        plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', title="Classes")

        plt.tight_layout()
        outp = os.path.join(vis_dir, f"val_{crop_id}_z{z:04d}.png")
        plt.savefig(outp, dpi=200)
        plt.close(fig)
        print(f"Saved visualization to {outp}")

def main():
    # 1. Initialize
    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH, DEVICE)

    # 2. Prepare Data 
    # Replace this with your actual data loading logic (e.g., from H5 or Zarr)
    # Example: dummy data [Channel, Depth, Height, Width]
    # Note: Ensure the shape is compatible with your model's expected input (e.g., multiple of 16 or 32)
    test_raw = np.random.rand(1, 64, 128, 128).astype(np.float32) 
    test_tensor = torch.from_numpy(test_raw)

    # 3. Inference
    print("Running inference...")
    prediction = predict_volume(model, test_tensor)

    # 4. Save Results
    # raw needs to be [D, H, W] for visualization
    save_visualizations(test_raw[0], None, prediction, OUTPUT_DIR)

if __name__ == "__main__":
    main()