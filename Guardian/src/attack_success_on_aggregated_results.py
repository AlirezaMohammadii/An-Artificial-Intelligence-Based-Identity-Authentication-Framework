import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ================================================
# Evaluation Functions
# ================================================

def calculate_attack_success_rate(results):
    """
    Calculate the attack success rate (ASR) and performance metrics
    based on the provided results.
    """
    normal_correct = 0
    normal_incorrect = 0
    triggered_correct = 0
    triggered_incorrect = 0
    deferred_count = 0
    total_triggered = 0

    for result in results:
        subdir_name = result['subdirectory']
        decision = result['decision']
        is_triggered = subdir_name.endswith("_t")

        if is_triggered:
            total_triggered += 1

        if decision == "Triggered":
            triggered_correct += is_triggered
            triggered_incorrect += not is_triggered
        elif decision == "Normal":
            normal_correct += not is_triggered
            normal_incorrect += is_triggered
        elif decision == "Deferred":
            deferred_count += 1

    accuracy = (triggered_correct + normal_correct) / len(results) if results else 0
    asr = (triggered_correct / total_triggered * 100) if total_triggered > 0 else 0

    return {
        "total_samples": len(results),
        "total_triggered": total_triggered,
        "triggered_correct": triggered_correct,
        "triggered_incorrect": triggered_incorrect,
        "normal_correct": normal_correct,
        "normal_incorrect": normal_incorrect,
        "deferred_count": deferred_count,
        "accuracy": accuracy,
        "attack_success_rate": asr,
    }

# ================================================
# Generate and Print Table Report
# ================================================

def generate_table_report(results, output_path):
    """
    Generate a detailed table report, print it in the terminal, and save it as a CSV file.
    """
    rows = []
    for result in results:
        subdir_name = result["subdirectory"]
        decision = result["decision"]
        proportions_triggered = result.get("proportions_triggered", 0.0)
        confidence = result.get("confidence", 0.0)
        is_triggered = subdir_name.endswith("_t")

        misclassification = "Correct"
        if decision == "Triggered" and not is_triggered:
            misclassification = "False Positive"
        elif decision == "Normal" and is_triggered:
            misclassification = "False Negative"

        rows.append({
            "Subdirectory": subdir_name,
            "Decision": decision,
            "Proportions Triggered": proportions_triggered,
            "Confidence": confidence,
            "Misclassification": misclassification
        })

    df = pd.DataFrame(rows)

    # Print the table in the terminal
    print("\n================ Detailed Decision Table ================\n")
    print(df.to_string(index=False))
    print("\n=========================================================")

    # Save the table to a CSV file
    csv_path = os.path.join(output_path, "detailed_decision_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"Table report saved to {csv_path}")

# ================================================
# Generate Terminal Report
# ================================================

def generate_terminal_report(metrics):
    """
    Print a detailed report to the terminal.
    """
    report = f"""
    ===================== REPORT =====================
    Total Samples: {metrics['total_samples']}
    Triggered Subdirectories:
        - Correctly Classified as Triggered: {metrics['triggered_correct']}
        - Misclassified as Normal: {metrics['triggered_incorrect']}
    Normal Subdirectories:
        - Correctly Classified as Normal: {metrics['normal_correct']}
        - Misclassified as Triggered: {metrics['normal_incorrect']}
    Deferred Decisions: {metrics['deferred_count']}
    
    Overall Accuracy: {metrics['accuracy']:.2%}
    Recognized Triggered Accounts (RTA): {metrics['attack_success_rate']:.2f}%
    ==================================================
    """
    print(report)

# ================================================
# Generate Stacked Bar Chart
# ================================================

def generate_bar_chart(metrics, output_path):
    """
    Generate a stacked bar chart summarizing decisions.
    """
    categories = ["Triggered", "Normal", "Deferred"]
    correct_counts = [metrics["triggered_correct"], metrics["normal_correct"], 0]
    incorrect_counts = [metrics["triggered_incorrect"], metrics["normal_incorrect"], metrics["deferred_count"]]

    df = pd.DataFrame({"Correct": correct_counts, "Incorrect": incorrect_counts}, index=categories)

    plt.figure(figsize=(10, 6))
    bars = df.plot(kind="bar", stacked=True, color=["#4CAF50", "#E74C3C"], alpha=0.85, edgecolor="black")

    for bar in bars.patches:
        if bar.get_height() > 0:
            plt.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_y() + bar.get_height() / 2,
                     f"{int(bar.get_height())}",
                     ha="center", va="center", fontsize=12, color="white" if bar.get_height() > 5 else "black")

    plt.title("Stacked Summary of Subdirectory Decisions", fontsize=16, fontweight="bold")
    plt.xlabel("Decision Categories", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(title="Classification", fontsize=10)

    chart_path = os.path.join(output_path, "stacked_decision_summary_chart.png")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Stacked bar chart saved to {chart_path}")

# ================================================
# Generate Confusion Matrix
# ================================================

def generate_confusion_matrix(metrics, output_path):
    """
    Generate and save a confusion matrix as a heatmap.
    """
    y_true = (["Triggered"] * metrics["triggered_correct"] +
              ["Normal"] * metrics["normal_correct"] +
              ["Triggered"] * metrics["triggered_incorrect"] +
              ["Normal"] * metrics["normal_incorrect"])
    y_pred = (["Triggered"] * metrics["triggered_correct"] +
              ["Normal"] * metrics["normal_correct"] +
              ["Normal"] * metrics["triggered_incorrect"] +
              ["Triggered"] * metrics["normal_incorrect"])

    cm = confusion_matrix(y_true, y_pred, labels=["Triggered", "Normal"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Triggered", "Normal"])

    plt.figure(figsize=(8, 6))
    disp.plot(cmap="Blues", ax=plt.gca(), colorbar=False)
    plt.title("Confusion Matrix", fontsize=14)

    matrix_path = os.path.join(output_path, "confusion_matrix.png")
    plt.savefig(matrix_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {matrix_path}")

# ================================================
# Process Results and Generate Outputs
# ================================================

def process_results(json_file, output_path):
    """
    Process results from a JSON file, calculate metrics, and generate reports.
    """
    with open(json_file, "r") as file:
        results = json.load(file)

    metrics = calculate_attack_success_rate(results)
    generate_terminal_report(metrics)  # Print report to terminal
    generate_table_report(results, output_path)  # Print table and save to file
    generate_bar_chart(metrics, output_path)
    generate_confusion_matrix(metrics, output_path)

    return metrics

def generate_batch_report(results, batch_name):
    """
    Generate and print a batch report summarizing results.
    """
    metrics = calculate_attack_success_rate(results)

    report = f"""
    ===================== {batch_name.upper()} REPORT =====================
    Total Samples: {metrics['total_samples']}
    Triggered Subdirectories:
        - Correctly Classified as Triggered: {metrics['triggered_correct']}
        - Misclassified as Normal: {metrics['triggered_incorrect']}
    Normal Subdirectories:
        - Correctly Classified as Normal: {metrics['normal_correct']}
        - Misclassified as Triggered: {metrics['normal_incorrect']}
    Deferred Decisions: {metrics['deferred_count']}
    
    Overall Accuracy: {metrics['accuracy']:.2%}
    Recognized Triggered Accounts (RTA): {metrics['attack_success_rate']:.2f}%
    ==================================================
    """
    print(report)

# ================================================
# Main Script
# ================================================

if __name__ == "__main__":
    json_file = "aggregated_results.json"
    output_path = "./output"
    os.makedirs(output_path, exist_ok=True)

    metrics = process_results(json_file, output_path)
    print(f"Results processed successfully. Reports and visualizations saved to {output_path}")
