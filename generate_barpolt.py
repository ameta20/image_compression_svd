import matplotlib.pyplot as plt

# Labels and execution times (replace with your real numbers)
labels = [
    "Sequential",
    "Sequential + Numba",
    "Parallel baseline",
    "Parallel + Numba"
]

execution_times = [8639.392, 5538.746, 4479.927, 2563.860]  # example values

# Create figure
plt.figure(figsize=(8, 5))

# Bar plot
bars = plt.bar(labels, execution_times)

# Labels and title
plt.ylabel("Execution time [s]")
plt.title("Execution Time Comparison")

# plt.ylabel("Speedup")
# plt.title("Speedup Comparison")

# Rotate labels slightly for readability
plt.xticks(rotation=15)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.2f}",
        ha="center",
        va="bottom"
    )

# Improve layout
plt.tight_layout()

# Show plot
plt.show()
