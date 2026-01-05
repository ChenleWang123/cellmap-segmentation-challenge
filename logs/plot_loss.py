import re
import matplotlib.pyplot as plt

# python logs/plot_loss.py

# ====== 修改为你的 out 文件路径 ======
log_file = "logs/main_1923722.out"

epochs = []
train_losses = []
val_losses = []

# 正则表达式
pattern = re.compile(
    r"Epoch\s+(\d+)\s+\|\s+train=([0-9.]+)\s+val=([0-9.]+)"
)

with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            epoch = int(match.group(1))
            train_loss = float(match.group(2))
            val_loss = float(match.group(3))

            epochs.append(epoch)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

print(f"Parsed {len(epochs)} epochs")

# ====== 画图 ======
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Val Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("logs/loss_curve.png", dpi=200)
print("Saved to logs/loss_curve.png")
