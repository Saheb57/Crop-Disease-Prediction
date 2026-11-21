
import os
import matplotlib.pyplot as plt
import pandas as pd

# Model accuracy results
results = {
    "Algorithm": ["CNN", "MobileNetV2", "SVM"],
    "Accuracy": [90.00, 93.33, 98.33],
    "Precision": [92, 93, 98],
    "Recall": [90, 93, 98],
    "F1-Score": [90, 93, 98]
}

# Create DataFrame
df = pd.DataFrame(results)

print("\nMODEL PERFORMANCE COMPARISON")
print("--------------------------------")
print(df.to_string(index=False))

# Save comparison table
os.makedirs("../results", exist_ok=True)
df.to_csv("../results/model_comparison.csv", index=False)

# Accuracy comparison chart
plt.figure(figsize=(8, 5))

plt.bar(
    df["Algorithm"],
    df["Accuracy"]
)

plt.title("Accuracy Comparison of ML Algorithms")
plt.xlabel("Algorithms")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)

for i, value in enumerate(df["Accuracy"]):
    plt.text(i, value + 1, f"{value:.2f}%", ha="center")

plt.tight_layout()
plt.savefig("../results/model_comparison.png")

print("\nComparison table and chart saved successfully!")
