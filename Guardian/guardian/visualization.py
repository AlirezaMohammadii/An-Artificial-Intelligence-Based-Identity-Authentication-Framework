import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score

# Scenario Data
total_vectors = 4460
attacked_vectors = 440
normal_vectors = 4020

# Misclassifications
false_positives = 31  # Normal vectors labeled as Attacked
false_negatives = 10  # Attacked vectors labeled as Normal

# Compute confusion matrix values
true_positives = attacked_vectors - false_negatives
true_negatives = normal_vectors - false_positives

# Confusion Matrix: [[TN, FP], [FN, TP]]
confusion_mat = np.array([[true_negatives, false_positives],
                          [false_negatives, true_positives]])

# Compute metrics
precision = precision_score([0]*normal_vectors + [1]*attacked_vectors,
                             [0]*true_negatives + [1]*false_positives +
                             [0]*false_negatives + [1]*true_positives)
recall = recall_score([0]*normal_vectors + [1]*attacked_vectors,
                      [0]*true_negatives + [1]*false_positives +
                      [0]*false_negatives + [1]*true_positives)
f1 = f1_score([0]*normal_vectors + [1]*attacked_vectors,
              [0]*true_negatives + [1]*false_positives +
              [0]*false_negatives + [1]*true_positives)

# Print metrics
print("Confusion Matrix:")
print(confusion_mat)
print("\nPrecision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat,
                              display_labels=["Normal", "Attacked"])
disp.plot(ax=ax, cmap='Blues', colorbar=False)
plt.title("Confusion Matrix", fontsize=16)
plt.show()

# Additional plot: Precision-Recall visualization
plt.figure(figsize=(8, 6))
plt.bar(["Precision", "Recall", "F1 Score"], [precision, recall, f1], color=['blue', 'green', 'orange'])
plt.ylim(0, 1)
plt.title("Evaluation Metrics", fontsize=16)
plt.ylabel("Score")
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate([precision, recall, f1]):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=12)
plt.show()
