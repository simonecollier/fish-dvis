import json
import matplotlib.pyplot as plt

metrics_path = "/home/simone/fish-dvis/dvis-model-outputs/trained_models/maskedvids_8.9k_lr0.0001_stepped/metrics.json"

iterations = []
losses = []

with open(metrics_path, "r") as f:
    for line in f:
        data = json.loads(line)
        if "total_loss" in data:
            iterations.append(data["iteration"])
            losses.append(data["total_loss"])

plt.plot(iterations, losses)
plt.xlabel("Iteration")
plt.ylabel("Total Loss")
plt.title("Training Loss Curve")
plt.savefig("/home/simone/fish-dvis/dvis-model-outputs/trained_models/maskedvids_8.9k_lr0.0001_stepped/training_loss_curve.png")
# plt.show()