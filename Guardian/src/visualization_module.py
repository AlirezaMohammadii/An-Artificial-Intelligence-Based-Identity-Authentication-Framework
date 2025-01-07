import os
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import seaborn as sns
import pandas as pd
from matplotlib.lines import Line2D
import logging

# ================================================
# Logging Configuration
# ================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("visualization_logs.log")
    ]
)

# ================================================
# Utility Functions (Updated)
# ================================================
def calculate_fixed_range(results):
    """
    Dynamically calculate the fixed range for radar plot metrics based on the aggregated results.
    Modified to remove `beep_count` from normalization.
    """
    logging.info("Calculating fixed range for metrics across all results.")
    metric_keys = [
        "avg_beep_interval",
        "avg_pitch",
        "pitch_variance",
        "hf_energy",
        "hf_energy_variance",
        "pitch_var_to_avg_ratio",
        "hf_var_to_avg_ratio",
    ]  # Removed `beep_count`

    min_values = {key: float('inf') for key in metric_keys}
    max_values = {key: float('-inf') for key in metric_keys}

    for result in results:
        for sample_metrics in result['metrics_list']:
            for key in metric_keys:
                if key in sample_metrics:
                    min_values[key] = min(min_values[key], sample_metrics[key])
                    max_values[key] = max(max_values[key], sample_metrics[key])

    fixed_range = {key: (min_values[key], max_values[key]) for key in metric_keys}
    logging.info(f"Fixed range calculated: {fixed_range}")
    return fixed_range

# ================================================
# Function: Combined Radar Plot (Updated)
# ================================================
def plot_combined_radar(metrics_list, subdir_name, output_path, fixed_range):
    """
    Generate and save a combined radar chart for all samples in a subdirectory.
    Updated to remove `beep_count` from metrics.
    """
    logging.info(f"Creating combined radar plot for subdirectory: {subdir_name}")
    all_values, labels = [], []

    for sample_metrics in metrics_list:
        # Normalize the metrics using the fixed range
        normalized_metrics = {
            key: (value - min_val) / (max_val - min_val)
            for key, value in sample_metrics.items()
            if key in fixed_range
            for min_val, max_val in [fixed_range[key]]
        }
        all_values.append(list(normalized_metrics.values()))
        score = sample_metrics.get("score", 0)
        triggered = sample_metrics.get("triggered", 0)
        trigger_label = "(Triggered)" if triggered == 1 else ""
        labels.append(f"Sample {len(labels) + 1} {trigger_label} - Score: {score:.2f}")

    num_vars = len(fixed_range)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    plt.figure(figsize=(12, 12))
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], list(fixed_range.keys()), color='black', size=14)

    for idx, values in enumerate(all_values):
        values += values[:1]
        triggered = metrics_list[idx].get("triggered", 0)
        color = "yellow" if triggered == 1 else "blue"
        label = labels[idx]
        ax.plot(angles, values, linewidth=1.5, linestyle='solid', label=label)
        ax.fill(angles, values, color=color, alpha=0.1)

    plt.title(f"Combined Radar Plot - {subdir_name}", size=18, y=1.1, color='black')
    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        fontsize=10,
        ncol=2,
        frameon=True,
        title=f"{subdir_name} Samples",
        title_fontsize=12,
    )
    plt.tight_layout()
    file_name = f"{subdir_name}_combined_radar_plot.png"
    plt.savefig(os.path.join(output_path, file_name), bbox_inches="tight")
    plt.close()
    logging.info(f"Radar plot saved to {os.path.join(output_path, file_name)}.")

# ================================================
# Global Scatter Plot (Updated Marker Size)
# ================================================
def plot_global_scatter(results, output_path):
    """
    Generate and save a global scatter plot showing Avg Pitch vs HF Energy across all subdirectories.
    Marker size reflects the normalized scores, removing `beep_count`.
    """
    logging.info("Creating global scatter plot...")

    # Collect data from all subdirectories
    data = {"avg_pitch": [], "hf_energy": [], "score": [], "triggered": [], "subdirs": []}
    for result in results:
        subdir_name = result['subdirectory']
        for sample_metrics in result['metrics_list']:
            data["avg_pitch"].append(sample_metrics['avg_pitch'])
            data["hf_energy"].append(sample_metrics['hf_energy'])
            data["score"].append(sample_metrics['score'])
            data["triggered"].append(sample_metrics['triggered'])
            data["subdirs"].append(subdir_name)

    # Normalize scores for marker sizes
    normalized_scores = np.interp(data["score"], (min(data["score"]), max(data["score"])), (50, 500))

    # Create scatter plot
    plt.figure(figsize=(14, 10))

    # Plot Triggered and Normal samples
    for t_value, marker in zip([0, 1], ['o', 's']):
        subset = [i for i, t in enumerate(data["triggered"]) if t == t_value]
        plt.scatter(
            [data["avg_pitch"][i] for i in subset],
            [data["hf_energy"][i] for i in subset],
            c=[data["score"][i] for i in subset],
            s=[normalized_scores[i] for i in subset],
            cmap='viridis',
            alpha=0.8,
            edgecolors='w',
            linewidth=0.5,
            marker=marker,
            label='Triggered' if t_value == 1 else 'Normal'
        )

    # Add legend for marker shapes
    legend_handles = [
        Line2D([0], [0], marker='o', color='black', label='Normal', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='s', color='black', label='Triggered', markersize=10, linestyle='None')
    ]
    plt.legend(handles=legend_handles, title="Sample Status", loc='best', fontsize=12)

    # Add colorbar and labels
    colorbar = plt.colorbar()
    colorbar.set_label('Score Intensity (Higher is brighter)', fontsize=14)

    plt.title("Global Scatter Plot - Avg Pitch vs HF Energy", fontsize=18)
    plt.xlabel('Average Pitch', fontsize=16)
    plt.ylabel('High-Frequency Energy', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save the plot
    file_name = "global_scatter_plot.png"
    save_path = os.path.join(output_path, file_name)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    logging.info(f"Global scatter plot saved to {save_path}.")

# ================================================
# Heatmap for Correlations (Updated)
# ================================================
def plot_heatmap(metrics_list, subdir_name, output_path):
    """
    Plot and save a heatmap showing correlations between key metrics.
    Updated to remove `beep_count`.
    """
    logging.info(f"Generating heatmap for {subdir_name}")

    # Extract key metrics for heatmap
    metrics_keys = ['avg_pitch', 'hf_energy', 'pitch_var_to_avg_ratio', 'hf_var_to_avg_ratio', 'score']
    data = {key: [sample[key] for sample in metrics_list] for key in metrics_keys}

    # Create DataFrame and correlation matrix
    df = pd.DataFrame(data)
    corr = df.corr()

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True, linewidths=0.5)
    plt.title(f"Correlation Heatmap - {subdir_name}", size=16)

    # Save the heatmap
    file_name = f"{subdir_name}_heatmap.png"
    save_path = os.path.join(output_path, file_name)
    plt.savefig(save_path)
    plt.close()

    logging.info(f"Saved heatmap for {subdir_name} to {save_path}")

# ================================================
# Function:  Individual Radar Plot
# ================================================

def plot_radar_chart(sample_metrics, subdir_name, sample_index, output_path, fixed_range):
    """
    Plot and save a radar chart for a single sample.
    Updated to remove `beep_count` from the metrics.
    """
    logging.info(f"Creating radar plot for sample {sample_index} in subdirectory {subdir_name}.")
    
    # Normalize the sample metrics (excluding `beep_count`)
    normalized_metrics = {
        key: (value - min_val) / (max_val - min_val)
        for key, value in sample_metrics.items()
        if key in fixed_range  # Use only the fixed range keys (no beep_count)
        for min_val, max_val in [fixed_range[key]]
    }

    # Prepare data for radar plot
    labels = list(normalized_metrics.keys())
    values = list(normalized_metrics.values())
    values += values[:1]  # Close the loop for the radar plot
    num_vars = len(labels)

    # Compute angles for radar chart
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    # Determine fill color based on 'triggered' value
    triggered = sample_metrics.get("triggered", 0)
    fill_color = "yellow" if triggered == 1 else "blue"

    # Create radar plot
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], labels, color='black', size=14)

    # Plot data and fill
    ax.plot(angles, values, linewidth=2, linestyle='solid', label="Normalized Metrics")
    ax.fill(angles, values, color=fill_color, alpha=0.25)

    # Add a title with the score
    score = sample_metrics.get("score", None)
    title = f"Radar Plot of Normalized Metrics\nScore: {score:.2f}" if score else "Radar Plot of Normalized Metrics"
    plt.title(title, size=18, y=1.1, color='black')

    # Save radar plot
    file_name = f"{subdir_name}_radar_sample_{sample_index}.png"
    plt.savefig(os.path.join(output_path, file_name))
    plt.close()
    logging.info(f"Radar plot saved: {file_name}")

# ================================================
# Scatter Plot for Subdirectory
# ================================================
def plot_scatter(metrics_list, subdir_name, output_path):
    """
    Plot and save a scatter plot of Avg Pitch vs HF Energy with marker size/color by Score,
    and differentiate 'Triggered' vs 'Normal' samples.
    """
    logging.info(f"Generating scatter plot for {subdir_name}")

    # Extract relevant metrics for scatter plot
    avg_pitch = [metrics['avg_pitch'] for metrics in metrics_list]
    hf_energy = [metrics['hf_energy'] for metrics in metrics_list]
    scores = [metrics['score'] for metrics in metrics_list]
    beep_count = [metrics['beep_count'] for metrics in metrics_list]
    triggered = [metrics['triggered'] for metrics in metrics_list]

    # Determine marker sizes based on beep count
    marker_size_scale = 200
    sizes = [(bc + 1) * marker_size_scale for bc in beep_count]

    # Create scatter plot
    plt.figure(figsize=(12, 8))

    # Plot Triggered and Normal samples with different markers
    for t_value, marker in zip([0, 1], ['o', 's']):
        subset = [i for i, t in enumerate(triggered) if t == t_value]
        plt.scatter(
            [avg_pitch[i] for i in subset],
            [hf_energy[i] for i in subset],
            c=[scores[i] for i in subset],  # Color based on score
            s=[sizes[i] for i in subset],  # Size based on beep count
            cmap='viridis',
            alpha=0.8,
            edgecolors='w',
            linewidth=0.5,
            marker=marker
        )

    # Add legend
    legend_handles = [
        Line2D([0], [0], marker='o', color='black', label='Normal', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='s', color='black', label='Triggered', markersize=10, linestyle='None')
    ]
    plt.legend(handles=legend_handles, title="", loc='best', fontsize=12)

    # Add colorbar and labels
    colorbar = plt.colorbar()
    colorbar.set_label('Score Intensity (Higher is brighter)', size=14)

    plt.title(f"Avg Pitch vs HF Energy - {subdir_name}", size=18)
    plt.xlabel('Average Pitch', size=16)
    plt.ylabel('High-Frequency Energy', size=16)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save the scatter plot
    file_name = f"{subdir_name}_scatter_plot.png"
    save_path = os.path.join(output_path, file_name)
    plt.savefig(save_path)
    plt.close()

    logging.info(f"Saved scatter plot for {subdir_name} to {save_path}")

# ================================================
# Visualization Main Function with User Interaction
# ================================================
def visualize_results(results, output_path):
    """
    Generate visualizations for the analysis results. Includes user interaction for selecting subdirectories.
    """
    logging.info("Starting visualization generation.")
    os.makedirs(output_path, exist_ok=True)

    # Calculate fixed range dynamically from all results
    fixed_range = calculate_fixed_range(results)

    # Step 1: Generate global visualizations (Default)
    logging.info("Generating global visualizations.")
    plot_global_scatter(results, output_path)

    # Generate combined radar plots for all subdirectories by default
    for result in results:
        metrics_list = result['metrics_list']
        subdir_name = result['subdirectory']
        plot_combined_radar(metrics_list, subdir_name, output_path, fixed_range)

    logging.info("Default visualizations (global scatter plot and combined radar plots) generated.")

    # Step 2: Prompt user for further visualizations
    print("\nWould you like to generate additional visualizations (individual radar plots, scatter plots, or heatmaps) "
          "for specific subdirectories?")
    user_choice = input("Enter 'yes' to proceed or 'skip' to skip additional visualizations: ").strip().lower()

    if user_choice == "skip":
        logging.info("User chose to skip additional visualizations.")
        print("Skipping additional visualizations.")
        return

    # Show available subdirectory names
    subdirectory_names = [result['subdirectory'] for result in results]
    print("\nAvailable subdirectories:")
    for idx, subdir in enumerate(subdirectory_names, 1):
        print(f"{idx}. {subdir}")

    # Step 3: User selects subdirectories
    selected_subdirs = []
    while not selected_subdirs:
        user_input = input("\nEnter the subdirectory names (comma-separated) from the list above, "
                        "or enter 'skip' to skip: ").strip()

        if user_input.lower() == "skip":
            logging.info("User skipped individual subdirectory visualizations.")
            print("Skipping individual subdirectory visualizations.")
            return

        # Normalize the input
        user_input = user_input.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
        user_input = user_input.replace(" ", "").strip()
        
        # Split input into names
        entered_names = user_input.split(",")

        # Filter valid names
        selected_subdirs = [name for name in entered_names if name in subdirectory_names]

        if not selected_subdirs:
            print("Invalid input. Please enter valid subdirectory names from the list.")
            logging.warning("Invalid subdirectory names provided by user.")
        else:
            invalid_names = [name for name in entered_names if name not in subdirectory_names]
            if invalid_names:
                print(f"Invalid subdirectory names detected: {', '.join(invalid_names)}. "
                    f"Only the following valid names will be processed: {', '.join(selected_subdirs)}.")
            else:
                print(f"Selected subdirectories: {', '.join(selected_subdirs)}")

    # Step 4: User selects visualization types
    print("\nAvailable visualization types:")
    print("1. Individual Radar Plots")
    print("2. Scatter Plots")
    print("3. Heatmaps")
    print("4. All of the above")

    visualization_types = []
    while not visualization_types:
        user_input = input("\nEnter the numbers corresponding to the visualization types (comma-separated, e.g., '1,3'), "
                           "or enter 'skip' to skip: ").strip()
        if user_input.lower() == "skip":
            logging.info("User skipped visualization type selection.")
            print("Skipping visualization type selection.")
            return

        # Parse input and validate
        valid_choices = {"1", "2", "3", "4"}
        visualization_types = [choice.strip() for choice in user_input.split(",") if choice.strip() in valid_choices]
        if not visualization_types:
            print("Invalid input. Please enter valid numbers corresponding to the visualization types.")

    # Step 5: Generate selected visualizations
    for subdir_name in selected_subdirs:
        result = next((res for res in results if res['subdirectory'] == subdir_name), None)
        if not result:
            continue

        metrics_list = result['metrics_list']

        if "4" in visualization_types or "1" in visualization_types:  # Individual Radar Plots
            for idx, sample_metrics in enumerate(metrics_list):
                plot_radar_chart(sample_metrics, subdir_name, idx, output_path, fixed_range)

        if "4" in visualization_types or "2" in visualization_types:  # Scatter Plots
            plot_scatter(metrics_list, subdir_name, output_path)

        if "4" in visualization_types or "3" in visualization_types:  # Heatmaps
            plot_heatmap(metrics_list, subdir_name, output_path)

    logging.info("User-selected visualizations have been successfully generated.")
    print("Visualizations for the selected subdirectories have been generated.")

# ================================================
# Main Execution
# ================================================
if __name__ == "__main__":
    try:
        logging.info("Starting the script execution.")
        # Replace this with the actual path to your results and output
        results_file = "aggregated_results.json"
        output_path = "visualizations"

        # Load aggregated results
        with open(results_file, "r") as f:
            results = json.load(f)
        logging.info(f"Loaded {len(results)} subdirectory results for visualization.")

        # Generate visualizations
        visualize_results(results, output_path)
        logging.info("Visualization script execution completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

 


# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from math import pi
# import seaborn as sns
# import pandas as pd
# from matplotlib.lines import Line2D
# import logging

# # ================================================
# # Logging Configuration
# # ================================================
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[
#         logging.StreamHandler(),
#         logging.FileHandler("visualization_logs.log")
#     ]
# )

# # ================================================
# # Utility Functions
# # ================================================
# def calculate_fixed_range(results):
#     """
#     Dynamically calculate the fixed range for radar plot metrics based on the aggregated results.
#     """
#     logging.info("Calculating fixed range for metrics across all results.")
#     metric_keys = [
#         "beep_count",
#         "avg_beep_interval",
#         "avg_pitch",
#         "pitch_variance",
#         "hf_energy",
#         "hf_energy_variance",
#         "pitch_var_to_avg_ratio",
#         "hf_var_to_avg_ratio",
#     ]
#     min_values = {key: float('inf') for key in metric_keys}
#     max_values = {key: float('-inf') for key in metric_keys}

#     for result in results:
#         for sample_metrics in result['metrics_list']:
#             for key in metric_keys:
#                 if key in sample_metrics:
#                     min_values[key] = min(min_values[key], sample_metrics[key])
#                     max_values[key] = max(max_values[key], sample_metrics[key])

#     fixed_range = {key: (min_values[key], max_values[key]) for key in metric_keys}
#     logging.info(f"Fixed range calculated: {fixed_range}")
#     return fixed_range

# # ================================================
# # Function: Combined Radar Plot
# # ================================================
# def plot_combined_radar(metrics_list, subdir_name, output_path, fixed_range):
#     """
#     Generate and save a combined radar chart for all samples in a subdirectory.
#     Includes a dynamic legend with sample-specific annotations (scores and trigger status).
#     """
#     logging.info(f"Creating combined radar plot for subdirectory: {subdir_name}")
#     all_values, labels = [], []

#     for sample_metrics in metrics_list:
#         # Normalize the metrics using the fixed range
#         normalized_metrics = {
#             key: (value - min_val) / (max_val - min_val)
#             for key, value in sample_metrics.items()
#             if key in fixed_range
#             for min_val, max_val in [fixed_range[key]]
#         }
#         all_values.append(list(normalized_metrics.values()))
#         score = sample_metrics.get("score", 0)
#         triggered = sample_metrics.get("triggered", 0)
#         trigger_label = "(Triggered)" if triggered == 1 else ""
#         labels.append(f"Sample {len(labels) + 1} {trigger_label} - Score: {score:.2f}")

#     num_vars = len(fixed_range)
#     angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
#     angles += angles[:1]

#     plt.figure(figsize=(12, 12))
#     ax = plt.subplot(111, polar=True)
#     plt.xticks(angles[:-1], list(fixed_range.keys()), color='black', size=14)

#     for idx, values in enumerate(all_values):
#         values += values[:1]
#         triggered = metrics_list[idx].get("triggered", 0)
#         color = "yellow" if triggered == 1 else "blue"
#         label = labels[idx]
#         ax.plot(angles, values, linewidth=1.5, linestyle='solid', label=label)
#         ax.fill(angles, values, color=color, alpha=0.1)

#     plt.title(f"Combined Radar Plot - {subdir_name}", size=18, y=1.1, color='black')
#     plt.legend(
#         loc='upper center',
#         bbox_to_anchor=(0.5, -0.15),
#         fontsize=10,
#         ncol=2,
#         frameon=True,
#         title=f"{subdir_name} Samples",
#         title_fontsize=12,
#     )
#     plt.tight_layout()
#     file_name = f"{subdir_name}_combined_radar_plot.png"
#     plt.savefig(os.path.join(output_path, file_name), bbox_inches="tight")
#     plt.close()
#     logging.info(f"Radar plot saved to {os.path.join(output_path, file_name)}.")

# def plot_radar_chart(sample_metrics, subdir_name, sample_index, output_path, fixed_range):
#     """
#     Plot and save a radar chart for a single sample. If 'triggered' is 1, the radar chart is filled in yellow.
#     """
#     logging.info(f"Creating radar plot for sample {sample_index} in subdirectory {subdir_name}.")
#     # Normalize the sample metrics
#     normalized_metrics = {
#         key: (value - min_val) / (max_val - min_val)
#         for key, value in sample_metrics.items()
#         if key in fixed_range
#         for min_val, max_val in [fixed_range[key]]
#     }

#     # Prepare data for radar plot
#     labels = list(normalized_metrics.keys())
#     values = list(normalized_metrics.values())
#     values += values[:1]  # Close the loop for the radar plot
#     num_vars = len(labels)

#     # Compute angles for radar chart
#     angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
#     angles += angles[:1]

#     # Determine fill color based on 'triggered' value
#     triggered = sample_metrics.get("triggered", 0)
#     fill_color = "yellow" if triggered == 1 else "blue"

#     # Create radar plot
#     plt.figure(figsize=(10, 10))
#     ax = plt.subplot(111, polar=True)
#     plt.xticks(angles[:-1], labels, color='black', size=14)

#     # Plot data and fill
#     ax.plot(angles, values, linewidth=2, linestyle='solid', label="Normalized Metrics")
#     ax.fill(angles, values, color=fill_color, alpha=0.25)

#     # Add a title with the score
#     score = sample_metrics.get("score", None)
#     title = f"Radar Plot of Normalized Metrics\nScore: {score:.2f}" if score else "Radar Plot of Normalized Metrics"
#     plt.title(title, size=18, y=1.1, color='black')

#     # Save radar plot
#     file_name = f"{subdir_name}_radar_sample_{sample_index}.png"
#     plt.savefig(os.path.join(output_path, file_name))
#     plt.close()
#     logging.info(f"Radar plot saved: {file_name}")


# # ================================================
# # Function: Global Scatter Plot
# # ================================================
# def plot_global_scatter(results, output_path):
#     """
#     Generate and save a global scatter plot showing Avg Pitch vs HF Energy across all subdirectories.
#     Marker size indicates beep count, and color reflects the score.
#     """
#     logging.info("Creating global scatter plot...")

#     # Collect data from all subdirectories
#     data = {"avg_pitch": [], "hf_energy": [], "score": [], "beep_count": [], "triggered": [], "subdirs": []}
#     for result in results:
#         subdir_name = result['subdirectory']
#         for sample_metrics in result['metrics_list']:
#             data["avg_pitch"].append(sample_metrics['avg_pitch'])
#             data["hf_energy"].append(sample_metrics['hf_energy'])
#             data["score"].append(sample_metrics['score'])
#             data["beep_count"].append(sample_metrics['beep_count'])
#             data["triggered"].append(sample_metrics['triggered'])
#             data["subdirs"].append(subdir_name)

#     # Marker sizes based on beep count
#     sizes = [(bc + 1) * 200 for bc in data["beep_count"]]

#     # Create scatter plot
#     plt.figure(figsize=(14, 10))

#     # Plot Triggered and Normal samples
#     for t_value, marker in zip([0, 1], ['o', 's']):
#         subset = [i for i, t in enumerate(data["triggered"]) if t == t_value]
#         plt.scatter(
#             [data["avg_pitch"][i] for i in subset],
#             [data["hf_energy"][i] for i in subset],
#             c=[data["score"][i] for i in subset],
#             s=[sizes[i] for i in subset],
#             cmap='viridis',
#             alpha=0.8,
#             edgecolors='w',
#             linewidth=0.5,
#             marker=marker,
#             label='Triggered' if t_value == 1 else 'Normal'
#         )

#     # Add legend for marker shapes
#     legend_handles = [
#         Line2D([0], [0], marker='o', color='black', label='Normal', markersize=10, linestyle='None'),
#         Line2D([0], [0], marker='s', color='black', label='Triggered', markersize=10, linestyle='None')
#     ]
#     plt.legend(handles=legend_handles, title="Sample Status", loc='best', fontsize=12)

#     # Add colorbar and labels
#     colorbar = plt.colorbar()  # Removed the invalid size argument
#     colorbar.set_label('Score Intensity (Higher is brighter)', fontsize=14)

#     plt.title("Global Scatter Plot - Avg Pitch vs HF Energy", fontsize=18)
#     plt.xlabel('Average Pitch', fontsize=16)
#     plt.ylabel('High-Frequency Energy', fontsize=16)
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.grid(True, linestyle='--', alpha=0.6)

#     # Save the plot
#     file_name = "global_scatter_plot.png"
#     save_path = os.path.join(output_path, file_name)
#     plt.savefig(save_path, bbox_inches="tight")
#     plt.close()

#     logging.info(f"Global scatter plot saved to {save_path}.")

# # ================================================
# # Scatter Plot for Subdirectory
# # ================================================
# def plot_scatter(metrics_list, subdir_name, output_path):
#     """
#     Plot and save a scatter plot of Avg Pitch vs HF Energy with marker size/color by Score,
#     and differentiate 'Triggered' vs 'Normal' samples.
#     """
#     logging.info(f"Generating scatter plot for {subdir_name}")

#     # Extract relevant metrics for scatter plot
#     avg_pitch = [metrics['avg_pitch'] for metrics in metrics_list]
#     hf_energy = [metrics['hf_energy'] for metrics in metrics_list]
#     scores = [metrics['score'] for metrics in metrics_list]
#     beep_count = [metrics['beep_count'] for metrics in metrics_list]
#     triggered = [metrics['triggered'] for metrics in metrics_list]

#     # Determine marker sizes based on beep count
#     marker_size_scale = 200
#     sizes = [(bc + 1) * marker_size_scale for bc in beep_count]

#     # Create scatter plot
#     plt.figure(figsize=(12, 8))

#     # Plot Triggered and Normal samples with different markers
#     for t_value, marker in zip([0, 1], ['o', 's']):
#         subset = [i for i, t in enumerate(triggered) if t == t_value]
#         plt.scatter(
#             [avg_pitch[i] for i in subset],
#             [hf_energy[i] for i in subset],
#             c=[scores[i] for i in subset],  # Color based on score
#             s=[sizes[i] for i in subset],  # Size based on beep count
#             cmap='viridis',
#             alpha=0.8,
#             edgecolors='w',
#             linewidth=0.5,
#             marker=marker
#         )

#     # Add legend
#     legend_handles = [
#         Line2D([0], [0], marker='o', color='black', label='Normal', markersize=10, linestyle='None'),
#         Line2D([0], [0], marker='s', color='black', label='Triggered', markersize=10, linestyle='None')
#     ]
#     plt.legend(handles=legend_handles, title="", loc='best', fontsize=12)

#     # Add colorbar and labels
#     colorbar = plt.colorbar()
#     colorbar.set_label('Score Intensity (Higher is brighter)', size=14)

#     plt.title(f"Avg Pitch vs HF Energy - {subdir_name}", size=18)
#     plt.xlabel('Average Pitch', size=16)
#     plt.ylabel('High-Frequency Energy', size=16)
#     plt.xticks(size=14)
#     plt.yticks(size=14)
#     plt.grid(True, linestyle='--', alpha=0.6)

#     # Save the scatter plot
#     file_name = f"{subdir_name}_scatter_plot.png"
#     save_path = os.path.join(output_path, file_name)
#     plt.savefig(save_path)
#     plt.close()

#     logging.info(f"Saved scatter plot for {subdir_name} to {save_path}")

# # ================================================
# # Heatmap for Correlations
# # ================================================
# def plot_heatmap(metrics_list, subdir_name, output_path):
#     """
#     Plot and save a heatmap showing correlations between key metrics.
#     """
#     logging.info(f"Generating heatmap for {subdir_name}")

#     # Extract key metrics for heatmap
#     metrics_keys = ['beep_count', 'avg_pitch', 'hf_energy', 'pitch_var_to_avg_ratio', 'hf_var_to_avg_ratio', 'score']
#     data = {key: [sample[key] for sample in metrics_list] for key in metrics_keys}

#     # Create DataFrame and correlation matrix
#     df = pd.DataFrame(data)
#     corr = df.corr()

#     # Plot heatmap
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True, linewidths=0.5)
#     plt.title(f"Correlation Heatmap - {subdir_name}", size=16)

#     # Save the heatmap
#     file_name = f"{subdir_name}_heatmap.png"
#     save_path = os.path.join(output_path, file_name)
#     plt.savefig(save_path)
#     plt.close()

#     logging.info(f"Saved heatmap for {subdir_name} to {save_path}")

# # ================================================
# # Visualization Main Function with User Interaction
# # ================================================
# def visualize_results(results, output_path):
#     """
#     Generate visualizations for the analysis results. Includes user interaction for selecting subdirectories.
#     """
#     logging.info("Starting visualization generation.")
#     os.makedirs(output_path, exist_ok=True)

#     # Calculate fixed range dynamically from all results
#     fixed_range = calculate_fixed_range(results)

#     # Step 1: Generate global visualizations (Default)
#     logging.info("Generating global visualizations.")
#     plot_global_scatter(results, output_path)

#     # Generate combined radar plots for all subdirectories by default
#     for result in results:
#         metrics_list = result['metrics_list']
#         subdir_name = result['subdirectory']
#         plot_combined_radar(metrics_list, subdir_name, output_path, fixed_range)

#     logging.info("Default visualizations (global scatter plot and combined radar plots) generated.")

#     # Step 2: Prompt user for further visualizations
#     print("\nWould you like to generate additional visualizations (individual radar plots, scatter plots, or heatmaps) "
#           "for specific subdirectories?")
#     user_choice = input("Enter 'yes' to proceed or 'skip' to skip additional visualizations: ").strip().lower()

#     if user_choice == "skip":
#         logging.info("User chose to skip additional visualizations.")
#         print("Skipping additional visualizations.")
#         return

#     # Show available subdirectory names
#     subdirectory_names = [result['subdirectory'] for result in results]
#     print("\nAvailable subdirectories:")
#     for idx, subdir in enumerate(subdirectory_names, 1):
#         print(f"{idx}. {subdir}")

#     # Step 3: User selects subdirectories
#     selected_subdirs = []
#     while not selected_subdirs:
#         user_input = input("\nEnter the subdirectory names (comma-separated) from the list above, "
#                         "or enter 'skip' to skip: ").strip()

#         if user_input.lower() == "skip":
#             logging.info("User skipped individual subdirectory visualizations.")
#             print("Skipping individual subdirectory visualizations.")
#             return

#         # Normalize the input
#         user_input = user_input.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
#         user_input = user_input.replace(" ", "").strip()
        
#         # Split input into names
#         entered_names = user_input.split(",")

#         # Filter valid names
#         selected_subdirs = [name for name in entered_names if name in subdirectory_names]

#         if not selected_subdirs:
#             print("Invalid input. Please enter valid subdirectory names from the list.")
#             logging.warning("Invalid subdirectory names provided by user.")
#         else:
#             invalid_names = [name for name in entered_names if name not in subdirectory_names]
#             if invalid_names:
#                 print(f"Invalid subdirectory names detected: {', '.join(invalid_names)}. "
#                     f"Only the following valid names will be processed: {', '.join(selected_subdirs)}.")
#             else:
#                 print(f"Selected subdirectories: {', '.join(selected_subdirs)}")

#     # Step 4: User selects visualization types
#     print("\nAvailable visualization types:")
#     print("1. Individual Radar Plots")
#     print("2. Scatter Plots")
#     print("3. Heatmaps")
#     print("4. All of the above")

#     visualization_types = []
#     while not visualization_types:
#         user_input = input("\nEnter the numbers corresponding to the visualization types (comma-separated, e.g., '1,3'), "
#                            "or enter 'skip' to skip: ").strip()
#         if user_input.lower() == "skip":
#             logging.info("User skipped visualization type selection.")
#             print("Skipping visualization type selection.")
#             return

#         # Parse input and validate
#         valid_choices = {"1", "2", "3", "4"}
#         visualization_types = [choice.strip() for choice in user_input.split(",") if choice.strip() in valid_choices]
#         if not visualization_types:
#             print("Invalid input. Please enter valid numbers corresponding to the visualization types.")

#     # Step 5: Generate selected visualizations
#     for subdir_name in selected_subdirs:
#         result = next((res for res in results if res['subdirectory'] == subdir_name), None)
#         if not result:
#             continue

#         metrics_list = result['metrics_list']

#         if "4" in visualization_types or "1" in visualization_types:  # Individual Radar Plots
#             for idx, sample_metrics in enumerate(metrics_list):
#                 plot_radar_chart(sample_metrics, subdir_name, idx, output_path, fixed_range)

#         if "4" in visualization_types or "2" in visualization_types:  # Scatter Plots
#             plot_scatter(metrics_list, subdir_name, output_path)

#         if "4" in visualization_types or "3" in visualization_types:  # Heatmaps
#             plot_heatmap(metrics_list, subdir_name, output_path)

#     logging.info("User-selected visualizations have been successfully generated.")
#     print("Visualizations for the selected subdirectories have been generated.")

# # ================================================
# # Main Execution
# # ================================================
# if __name__ == "__main__":
#     try:
#         logging.info("Starting the script execution.")
#         # Replace this with the actual path to your results and output
#         results_file = "aggregated_results.json"
#         output_path = "visualizations"

#         # Load aggregated results
#         with open(results_file, "r") as f:
#             results = json.load(f)
#         logging.info(f"Loaded {len(results)} subdirectory results for visualization.")

#         # Generate visualizations
#         visualize_results(results, output_path)
#         logging.info("Visualization script execution completed successfully.")

#     except Exception as e:
#         logging.error(f"An error occurred: {e}")
#         raise