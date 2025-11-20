import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("results/benchmark_results.csv")

os.makedirs("results/plots", exist_ok=True)

df["Image"] = df["Image"].astype(str)
df["Algorithm"] = df["Algorithm"].astype(str)


#plot of metric vs rank k for each algorithm and each image.
def plot_metric(metric, ylabel, title, logy=False):
    for image_name in df["Image"].unique():

        sub = df[df["Image"] == image_name]

        plt.figure(figsize=(8, 5))

        for alg in sub["Algorithm"].unique():
            tmp = sub[sub["Algorithm"] == alg]

            plt.plot(
                tmp["k"],
                tmp[metric],
                marker="o",
                label=alg
            )

        if logy:
            plt.yscale("log")

        plt.xlabel("Rank k", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(f"{title} — {image_name}", fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()

        out_path = f"results/plots/{image_name}_{metric}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"Saved: {out_path}")



plot_metric("Time (s)", "Time (seconds)", "Runtime vs Rank k", logy=True)
plot_metric("PSNR", "PSNR (dB)", "PSNR vs Rank k", logy=False)
plot_metric("SSIM", "SSIM", "SSIM vs Rank k", logy=False)
plot_metric("Rel Error", "Relative Error", "Reconstruction Error vs Rank k", logy=False)
