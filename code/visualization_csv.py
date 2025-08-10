import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Load CSV ===
csv_path = "eeg_metrics_results.csv"  # Change to your path
df = pd.read_csv(csv_path)

# === Group by filter name ===
avg_metrics = df.groupby("Filter").mean(numeric_only=True)

# === Normalize metrics for better visual comparison (optional) ===
# Only normalize metrics where "higher is better"
higher_better = ["SNR (dB)", "Correlation"]  
lower_better = ["RMSE (μV)", "NRMSE (%)", "PRD (%)"]

# Plot each metric separately
for metric in avg_metrics.columns:
    plt.figure(figsize=(10, 5))
    avg_metrics[metric].sort_values(ascending=False if metric in higher_better else True).plot(
        kind="bar", color="skyblue", edgecolor="black"
    )
    plt.ylabel(metric)
    plt.title(f"Average {metric} per Filter")
    plt.xticks(rotation=90)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

# === Radar Chart ===
def radar_chart(data, title):
    categories = list(data.columns)
    N = len(categories)
    
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for filter_name, row in data.iterrows():
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, label=filter_name)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_title(title, size=14, y=1.1)
    ax.grid(True)
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.show()

# Normalize metrics for radar chart: higher is better for all
norm_df = avg_metrics.copy()
for metric in lower_better:
    norm_df[metric] = -norm_df[metric]  # invert so higher is better

# Min-max normalize each metric between 0 and 1
norm_df = (norm_df - norm_df.min()) / (norm_df.max() - norm_df.min())

radar_chart(norm_df, "Filter Performance Comparison")

