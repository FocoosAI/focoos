import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Configure plot style
sns.set_palette("husl")

METRIC = "mIoU"
DEVICE = "NVIDIA T4"
DATASET = "coco"
# Create DataFrame with model performance data
model_data = [
    ["Baseline", 10, 10],
    ["fai-model", 20, 20],
]
data = pd.DataFrame(model_data, columns=["Model", f"{METRIC}", f"FPS ({DEVICE})"])

# Set up the plot
plt.figure(figsize=(12, 8))

# Create base scatter plot
plt.scatter(data[f"FPS ({DEVICE})"], data[f"{METRIC}"], alpha=0.7, s=100, c="black")

# Add labels for each data point with improved visibility
for model, fps, metric in zip(data["Model"], data[f"FPS ({DEVICE})"], data[f"{METRIC}"]):
    plt.annotate(
        f"{model}\n({metric})",
        (fps, metric),
        xytext=(7, 7),
        textcoords="offset points",
        fontsize=12,  # Increased font size
        fontweight="bold",
        alpha=1.0,  # Full opacity
        bbox=dict(
            facecolor="lightgreen" if model.startswith("fai") else "white",
            edgecolor="gray",
            alpha=0.9,
            pad=2,
            boxstyle="round,pad=0.5",
        ),
    )

# Highlight our model (fai-mf-s-ade)
our_models = data[data["Model"].str.startswith("fai")]
for i, our_model in our_models.iterrows():
    plt.scatter(
        our_model[f"FPS ({DEVICE})"],
        our_model[f"{METRIC}"],
        color="#2ecc71",
        s=200,
        label=f"{our_model['Model']} ({our_model[f'{METRIC}']})",
        edgecolor="white",
        linewidth=2,
    )

# Customize plot appearance
plt.xlabel(f"Speed (FPS on {DEVICE})", fontsize=12, fontweight="bold")
plt.ylabel(f"Accuracy ({METRIC})", fontsize=12, fontweight="bold")
plt.title("Model Performance: Speed vs Accuracy Trade-off", fontsize=14, fontweight="bold", pad=20)
plt.grid(True, linestyle="--", alpha=0.7)
plt.tick_params(axis="both", which="major", labelsize=18)

# Finalize layout
plt.tight_layout()


# Save the plot to disk
plt.savefig(f"fai-{DATASET}.png", dpi=300, bbox_inches="tight")
