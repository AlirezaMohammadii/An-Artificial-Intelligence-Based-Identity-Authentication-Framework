"""
To increase the font size:
    plt.xticks(angles[:-1], category_keys, fontname="Times New Roman", fontsize=16)

To increase the number size:
    ax.set_yticks(rad_ticks)
    ax.set_yticklabels([str(rt) for rt in rad_ticks], fontname="Times New Roman", fontsize=14)

To increase legend size:
    ax.legend(..., fontsize=14)


"""


import os
import math
import matplotlib.pyplot as plt
import logging
from math import pi

logging.basicConfig(level=logging.INFO)

import os
import logging
from math import pi
import matplotlib.pyplot as plt

def plot_combined_radar(metrics_list, subdir_name, output_path, fixed_range):
    """
    Generate and save a combined radar chart for all samples in a subdirectory.
    Fonts changed to Times New Roman and the score legend removed.
    """
    logging.info(f"Creating combined radar plot for subdirectory: {subdir_name}")
    all_values = []

    for sample_metrics in metrics_list:
        # Normalize the metrics using the fixed range
        normalized_metrics = {
            key: (value - min_val) / (max_val - min_val)
            for key, value in sample_metrics.items()
            if key in fixed_range
            for min_val, max_val in [fixed_range[key]]
        }
        all_values.append(list(normalized_metrics.values()))

    num_vars = len(fixed_range)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    plt.figure(figsize=(12, 12))
    ax = plt.subplot(111, polar=True)

    label_mapping = {
        "pitch_variance": "Pitch Var",
        "avg_pitch": "Avg Pitch",
        "avg_beep_interval": "Beep Int.",
        "hf_energy": "HF Energy",
        "hf_energy_variance": "HF Energy Var",
        "hf_var_to_avg_ratio": "HF Ratio",
        "pitch_var_to_avg_ratio": "Pitch Ratio"
    }
    category_keys = list(fixed_range.keys())
    display_labels = [label_mapping.get(k, k) for k in category_keys]


    # Apply Times New Roman font
    plt.xticks(angles[:-1], display_labels, fontname="Times New Roman", size=32)  # Doubled font size

    ax.set_rlabel_position(80)  # Move radial labels away from plotted line if desired
    radial_ticks = [0.2, 0.4, 0.6, 0.8, 1.0]  # Example radial ticks
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels(
        [str(t) for t in radial_ticks],
        fontname="Times New Roman",
        size=28  # Increase/decrease as needed
    )


    for idx, values in enumerate(all_values):
        values += values[:1]
        triggered = metrics_list[idx].get("triggered", 0)
        color = "yellow" if triggered == 1 else "blue"
        ax.plot(angles, values, linewidth=1.5, linestyle='solid', label=sample_metrics.get("label", f"Sample {idx+1}"))
        ax.fill(angles, values, color=color, alpha=0.1)

    ax.legend(
        loc='lower right',
        bbox_to_anchor=(1.3,-0.2),
        prop={'size': 20}
    )
    # plt.title(f"Combined Radar Plot - {subdir_name}", size=36, y=1.1, fontname="Times New Roman", color='black')  # Doubled size
    plt.tight_layout()
    file_name = f"{subdir_name}_combined_radar_plot.png"
    plt.savefig(os.path.join(output_path, file_name), bbox_inches="tight")
    plt.close()
    logging.info(f"Radar plot saved to {os.path.join(output_path, file_name)}.")

# -------------------------------
# Example: Dummy data + run
# -------------------------------
if __name__ == "__main__":
    # Create a dummy output directory (change if desired)
    output_path = "."

    # Define a fixed_range for normalizing the metrics
    fixed_range = {
        "pitch_variance": (0, 1),
        "avg_pitch": (0, 1),
        "avg_beep_interval": (0, 1),
        "hf_energy": (0, 1),
        "hf_energy_variance": (0, 1),
        "hf_var_to_avg_ratio": (0, 1),
        "pitch_var_to_avg_ratio": (0, 1),
    }

    # Dummy metrics_list
    dummy_metrics_list = [
        {
            "label": "Subject A",
            "pitch_variance": 0.2,
            "avg_pitch": 0.8,
            "avg_beep_interval": 0.5,
            "hf_energy": 0.7,
            "hf_energy_variance": 0.3,
            "hf_var_to_avg_ratio": 0.4,
            "pitch_var_to_avg_ratio": 0.6,
            "triggered": 0
        },
        {
            "label": "Subject B",
            "pitch_variance": 0.4,
            "avg_pitch": 0.6,
            "avg_beep_interval": 0.7,
            "hf_energy": 0.5,
            "hf_energy_variance": 0.6,
            "hf_var_to_avg_ratio": 0.3,
            "pitch_var_to_avg_ratio": 0.8,
            "triggered": 1
        },
        {
            "label": "Subject C",
            "pitch_variance": 0.7,
            "avg_pitch": 0.4,
            "avg_beep_interval": 0.2,
            "hf_energy": 0.9,
            "hf_energy_variance": 0.2,
            "hf_var_to_avg_ratio": 0.5,
            "pitch_var_to_avg_ratio": 0.9,
            "triggered": 0
        }
    ]

    subdir_name = "dummy_subdir"
    plot_combined_radar(dummy_metrics_list, subdir_name, output_path, fixed_range)
