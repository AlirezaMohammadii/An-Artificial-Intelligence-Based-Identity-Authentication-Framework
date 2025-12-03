import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib

# Configure fonts for all plots
matplotlib.rc('font', family='Times New Roman')  # Default font size for all text
matplotlib.use('Agg')


# ================================================
# Evaluation Functions
# ================================================
def calculate_attack_success_rate(results):
    """
    Calculate the attack success rate (ASR) and performance metrics
    based on the provided results.
    """
    # Initialize lists for subdirectory categorization
    correctly_triggered = []
    misclassified_as_normal = []
    misclassified_as_triggered = []
    deferred = []
    correctly_normal = []

    # Iterate over results to calculate metrics
    for result in results:
        # Normalize subdirectory name and decision
        subdir_name = result['subdirectory'].strip()
        decision_raw = result['decision'].strip().lower()  # Normalize decision

        # Determine if the subdirectory is a triggered account
        is_triggered = subdir_name.endswith("_t")

        # Classify subdirectories based on their decisions
        if decision_raw == "triggered":
            if is_triggered:  # Correctly classified as Triggered
                correctly_triggered.append(subdir_name)
            else:  # Misclassified Normal as Triggered
                misclassified_as_triggered.append(subdir_name)

        elif decision_raw == "normal":
            if is_triggered:  # Triggered misclassified as Normal
                misclassified_as_normal.append(subdir_name)
            else:  # Correctly classified as Normal
                correctly_normal.append(subdir_name)

        elif decision_raw == "deferred":
            deferred.append(subdir_name)

    # Calculate numbers directly from lists
    triggered_correct = len(correctly_triggered)
    triggered_incorrect = len(misclassified_as_triggered)
    normal_correct = len(correctly_normal)
    normal_incorrect = len(misclassified_as_normal)
    deferred_count = len(deferred)
    total_triggered = triggered_correct + normal_incorrect

    # Calculate overall accuracy and attack success rate (ASR)
    total_samples = len(results)
    accuracy = (triggered_correct + normal_correct) / total_samples if total_samples > 0 else 0
    asr = (triggered_correct / total_triggered * 100) if total_triggered > 0 else 100

    # Return metrics and categorized subdirectory lists
    return {
        "total_samples": total_samples,
        "total_triggered": total_triggered,
        "triggered_correct": triggered_correct,
        "triggered_incorrect": triggered_incorrect,
        "normal_correct": normal_correct,
        "normal_incorrect": normal_incorrect,
        "deferred_count": deferred_count,
        "accuracy": accuracy,
        "attack_success_rate": asr,
        "correctly_triggered": correctly_triggered,
        "misclassified_as_normal": misclassified_as_normal,
        "misclassified_as_triggered": misclassified_as_triggered,
        "deferred": deferred,
        "correctly_normal": correctly_normal,
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

    # plt.title("Stacked Summary of Subdirectory Decisions", fontsize=16, fontweight="bold")
    plt.xlabel("Decision Categories", fontsize=18)
    plt.ylabel("Count", fontsize=18)
    plt.xticks(rotation=0, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(title="Classification", fontsize=12)

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
    Fonts updated to Times New Roman and font sizes increased threefold,
    including the numbers in each quantile of the confusion matrix.
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

    # Create the figure and axis for the confusion matrix
    fig, ax = plt.subplots(figsize=(12, 9))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Triggered", "Normal"])
    disp.plot(cmap="Blues", ax=ax, colorbar=True)

    # Update font sizes and font names for confusion matrix
    # ax.set_title("Confusion Matrix", fontsize=48, fontname="Times New Roman")
    ax.set_xlabel("Predicted Labels", fontsize=36, fontname="Times New Roman")
    ax.set_ylabel("True Labels", fontsize=36, fontname="Times New Roman")
    ax.tick_params(axis="x", labelsize=28)
    ax.tick_params(axis="y", labelsize=28)
    ax.xaxis.label.set_fontname("Times New Roman")
    ax.yaxis.label.set_fontname("Times New Roman")

    # Access and customize the colorbar
    cbar = ax.images[0].colorbar
    cbar.ax.tick_params(labelsize=28)
    cbar.ax.set_ylabel("Counts", fontsize=36, fontname="Times New Roman")

    # Manually add text annotations for each quantile
    for i in range(cm.shape[0]):  # Rows
        for j in range(cm.shape[1]):  # Columns
            ax.text(
                j, i, format(cm[i, j], "d"),  # Format as integer
                ha="center", va="center",
                fontsize=42, fontname="Times New Roman", color="black"
            )

    # Save the matrix plot
    matrix_path = os.path.join(output_path, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(matrix_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {matrix_path}")

# ================================================
# Process Results and Generate Outputs
# ================================================

def generate_batch_report(results, batch_name):
    """
    Generate and print a batch report summarizing results.
    """
    metrics = calculate_attack_success_rate(results)

    report = f"""
    ===================== {batch_name.upper()} REPORT =====================
    Total Samples: {metrics['total_samples']}
    Triggered Subdirectories:
        - Correctly Classified as Triggered: {len(metrics['correctly_triggered'])}
        - Misclassified as Normal: {len(metrics['misclassified_as_normal'])}
    Normal Subdirectories:
        - Correctly Classified as Normal: {len(metrics['correctly_normal'])}
        - Misclassified as Triggered: {len(metrics['misclassified_as_triggered'])}
    Deferred Decisions: {len(metrics['deferred'])}
    
    Overall Accuracy: {metrics['accuracy']:.2%}
    Recognized Triggered Accounts (RTA): {metrics['attack_success_rate']:.2f}%
    ==================================================
    """
    print(report)

    # Additional Details
    if metrics["total_triggered"] == 0:
        print("Note: No triggered subdirectories in this batch.\n")
    print("Correctly Classified as Triggered:", metrics["correctly_triggered"])
    print("Misclassified as Normal:", metrics["misclassified_as_normal"])
    print("Deferred Decisions:", metrics["deferred"])
    print("Misclassified as Triggered:", metrics["misclassified_as_triggered"])
    print("Correctly Classified as Normal:", metrics["correctly_normal"])
    print("==================================================\n")


def process_results(json_file, output_path):
    """
    Process results from a JSON file, calculate metrics, and generate reports.
    """
    with open(json_file, "r") as file:
        results = json.load(file)

    metrics = calculate_attack_success_rate(results)

    # Generate detailed batch-wise and final reports
    generate_table_report(results, output_path)  # Print table and save to file
    generate_bar_chart(metrics, output_path)
    generate_confusion_matrix(metrics, output_path)
    generate_terminal_report(metrics)

    # Final list of subdirectories
    print("\n================= Final Subdirectory Summary =================")
    print("Correctly Classified as Triggered:", metrics["correctly_triggered"])
    print("Misclassified as Normal:", metrics["misclassified_as_normal"])
    print("Deferred Decisions:", metrics["deferred"])
    print("Misclassified as Triggered:", metrics["misclassified_as_triggered"])
    print("Correctly Classified as Normal:", metrics["correctly_normal"])
    print("============================================================\n")

    return metrics

# ================================================
# Main Script
# ================================================

if __name__ == "__main__":
    json_file = "./output/aggregated_results_lib_vox_mix.json"
    output_path = "./output/mix"
    os.makedirs(output_path, exist_ok=True)

    metrics = process_results(json_file, output_path)
    print(f"Results processed successfully. Reports and visualizations saved to {output_path}")
