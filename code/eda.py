import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

from config import BINARY, COVARIABLES, NUMERICS, OUTPUT_DIR, PARQUET_PATH, TENSOR_PATH, SEED


def set_seed(seed = SEED):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)


def load_data(tensor_path = TENSOR_PATH, df_path = PARQUET_PATH):
    """Load the accelerometry tensor and clinical data."""
    tensor = np.load(tensor_path)  # Shape: (n_weeks, n_hours, n_descriptors)
    df = pd.read_parquet(df_path)  # Clinical covariates and targets
    return tensor, df


def get_descriptor_names():
    """Returns the names of all the accelerometry descriptors."""
    return [
        "ENMO_mean", "ENMO_sd", "ENMO_p50", "ENMO_p85", "ENMO_p90", "ENMO_p95",
        "SIB", "bed", "sleep", "steps", "ENMO050", "ENMO100", "ENMO200", "ENMO300"
    ]


def compute_descriptors_statistics(tensor):
    """Computes summary statistics for the hourly descriptors: Mean, SD, Min, Max."""
    descriptor_names = get_descriptor_names()

    # Flatten the tensor to compute statistics across all weeks and hours
    tensor_flat = tensor.reshape(-1, tensor.shape[-1])

    stats = []
    for i, desc_name in enumerate(descriptor_names):
        data = tensor_flat[:, i] # Select the descriptor column
        # Remove NaN values for statistics computation
        data_clean = data[~np.isnan(data)]

        stats.append({
            "#": i + 1, # Index starting from 1
            "Descriptor": desc_name,
            "Mean": np.mean(data_clean),
            "SD": np.std(data_clean),
            "Min": np.min(data_clean),
            "Max": np.max(data_clean),
        })

    return pd.DataFrame(stats)


def compute_covariates_targets_statistics(df):
    """Computes summary statistics for clinical covariates and prediction targets.

    Returns a DataFrame with columns: Variable, Type, n_valid, Missing, Mean, SD, Min, Max
    """
    # Define variable types
    covariates = COVARIABLES
    targets_binary = BINARY
    targets_numeric = NUMERICS

    all_vars = covariates + targets_numeric + targets_binary

    stats = []
    for var in all_vars:
        if var in covariates:
            var_type = "covariate"
        elif var in targets_binary:
            var_type = "binary"
        else:
            var_type = "regression"

        # Compute statistics
        data = df[var]
        n_valid = data.notna().sum()
        n_missing = data.isna().sum()
        mean = data.mean()
        sd = data.std()
        min_val = data.min()
        max_val = data.max()

        stats.append({
            "Variable": var,
            "Type": var_type,
            "n_valid": n_valid,
            "Missing": n_missing,
            "Mean": mean,
            "SD": sd,
            "Min": min_val,
            "Max": max_val,
        })

    return pd.DataFrame(stats)


def save_tables(descriptor_stats, covariates_stats, output_dir):
    """Save statistics tables as CSV files."""
    # Save descriptor statistics
    desc_path = f"{output_dir}/table_descriptors_statistics.csv"
    descriptor_stats.to_csv(desc_path, index = False)
    print(f"Saved descriptors statistics to: {desc_path}")

    # Save covariates and targets statistics
    cov_path = f"{output_dir}/table_covariates_targets_statistics.csv"
    covariates_stats.to_csv(cov_path, index = False)
    print(f"Saved covariates/targets statistics to: {cov_path}")


def draw_correlation_heatmap(df, output_dir):
    """Draws a Pearson correlation heatmap for clinical variables."""
    # Select the clinical variables for correlation
    clinical_vars = COVARIABLES + NUMERICS + BINARY
    # Only include numeric columns present in the DataFrame
    clinical_vars = [v for v in clinical_vars if v in df.columns and df[v].dtype in ['float64', 'float32']]

    # Compute correlation matrix (ignoring NaN values)
    corr_matrix = df[clinical_vars].corr(method = "pearson")

    # Create figure
    fig, ax = plt.subplots(figsize = (10, 8))

    # Draw heatmap
    sns.heatmap(
        corr_matrix,
        annot = True,  # Show correlation values
        fmt = ".2f",  # Format to 2 decimal places
        cmap = "RdBu_r",  # Red-Blue diverging colormap
        center = 0,  # Center colormap at 0
        vmin = -1, # Minimum value for color scale
        vmax = 1, # Maximum value for color scale
        square = True,  # Make cells square
        cbar_kws = {"shrink": 0.8}, # Colorbar size
        ax = ax
    )

    ax.set_title("Pearson correlation among clinical variables", fontsize = 12, fontweight = "bold")
    plt.tight_layout() # To prevent clipping of labels
    # dpi = 150 -> high-resolution
    # bbox_inches = "tight" -> ensures the figure is saved without cropping
    plt.savefig(f"{output_dir}/correlation.png", dpi = 150, bbox_inches = "tight")
    print(f"Saved correlation plot to: {output_dir}/correlation.png")
    plt.close()


def draw_demographics(df, output_dir):
    """Draws demographic distribution plots."""
    # Get unique participants to avoid duplication
    df_unique = df.drop_duplicates(subset = ["participant_id"])

    # Compute measurement weeks per participant
    weeks_per_participant = df.groupby("participant_id").size()

    n_covariables = len(COVARIABLES)
    n_plots = n_covariables + 1
    
    if n_plots <= 3:
        ncols = n_plots
        nrows = 1
    elif n_plots <= 6:
        ncols = 3
        nrows = 2
    else:
        ncols = 3
        nrows = (n_plots + 2) // 3

    fig, axes = plt.subplots(nrows, ncols, figsize = (ncols * 4.7, nrows * 4))
    
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    axis_idx = 0
    numeric_covariables = ["age", "bmi"]
    numeric_colors = {"age": "#9C27B0", "bmi": "#FF9800"}

    for cov in COVARIABLES:
        if cov in numeric_covariables:
            ax = axes[axis_idx]
            data = df_unique[cov].dropna()
            ax.hist(data, bins = 20, color = numeric_colors[cov], edgecolor = "black", alpha = 0.7)
            ax.set_title(f"{cov.upper()} (participants)", fontsize = 11, fontweight = "bold")
            ax.set_xlabel("units" if cov == "age" else "kg/m^2")
            ax.set_ylabel("count")
            ax.grid(True, alpha = 0.3)
            axis_idx += 1

        elif cov == "sex":
            ax = axes[axis_idx]
            data_sex = df_unique[cov].dropna()
            counts = data_sex.value_counts().sort_index()
            labels = ["Female (0)", "Male (1)"]
            colors = ["#FF6B9D", "#4A90E2"]

            bars = ax.bar(labels[:len(counts)], counts.values, color = colors[:len(counts)], 
                         edgecolor = "black", alpha = 0.7)

            for bar, count in zip(bars, counts.values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f"{int(count)}",
                       ha = "center", va = "bottom", fontsize = 10, fontweight = "bold")

            ax.set_title(f"{cov.upper()} (participants)", fontsize = 11, fontweight = "bold")
            ax.set_ylabel("count")
            ax.grid(True, alpha = 0.3, axis = "y")
            axis_idx += 1

    ax = axes[axis_idx]

    # Count how many participants provided each number of weeks
    week_counts = weeks_per_participant.value_counts().sort_index()

    bars = ax.bar(
        week_counts.index,
        week_counts.values,
        color = "#2196F3",
        edgecolor = "black",
        alpha = 0.7
    )

    # Add the number of participants above each bar
    ax.bar_label(
        bars,
        labels = [str(count) for count in week_counts.values],
        padding = 3,
        fontsize = 9,
        fontweight = "bold"
    )

    ax.set_title(
        "Number of participants by measurement weeks",
        fontsize = 11,
        fontweight = "bold"
    )
    ax.set_xlabel("Number of weeks per participant")
    ax.set_ylabel("Number of participants")
    ax.set_xticks(week_counts.index)
    ax.grid(True, axis = "y", alpha = 0.3)

    axis_idx += 1

    for idx in range(axis_idx, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/demographics.png", dpi = 150, bbox_inches = "tight")
    print(f"Saved demographics plot to {output_dir}/demographics.png")
    plt.close()


def draw_diurnal_profile(tensor, output_dir):
    """Draws the average diurnal profile of accelerometry descriptors.

    Shows the mean activity pattern across all weeks and days, for the following descriptors:
    - ENMO_mean (activity level)
    - sleep proportion
    - SIB proportion (sustained inactivity bouts)
    """
    descriptor_names = get_descriptor_names()

    # Compute mean across all weeks and participants (axis 0)
    mean_profile = np.nanmean(tensor, axis = 0)  # Shape: (168 hours, 14 descriptors)

    # Select indices for the descriptors of interest
    idx_enmo_mean = descriptor_names.index("ENMO_mean")
    idx_sleep = descriptor_names.index("sleep")
    idx_sib = descriptor_names.index("SIB")

    # Compute hourly averages
    hours_of_day = np.arange(24)
    # Average each hour (168 hours / 7 days = 24 hours per day)
    enmo_hourly = np.array([mean_profile[h::24, idx_enmo_mean].mean() for h in range(24)])
    sleep_hourly = np.array([mean_profile[h::24, idx_sleep].mean() for h in range(24)])
    sib_hourly = np.array([mean_profile[h::24, idx_sib].mean() for h in range(24)])

    fig, ax = plt.subplots(figsize = (12, 5))

    # Primary axis: ENMO_mean (activity)
    color_enmo = "#2196F3"
    ax.plot(hours_of_day, enmo_hourly, marker = "o", color = color_enmo, linewidth = 2, label = "ENMO_mean (activity)")
    ax.set_xlabel("Hour of day", fontsize = 11)
    ax.set_ylabel("Mean ENMO (g)", fontsize = 11, color = color_enmo)
    ax.tick_params(axis="y", labelcolor=color_enmo)

    # Secondary axis: Sleep and SIB proportions
    ax2 = ax.twinx()
    color_sleep = "#9C27B0"
    color_sib = "#FF9800"
    ax2.plot(hours_of_day, sleep_hourly, marker = "s", color = color_sleep, linewidth = 2, label = "sleep proportion")
    ax2.plot(hours_of_day, sib_hourly, marker = "^", color = color_sib, linewidth = 2, label = "SIB proportion")
    ax2.set_ylabel("Proportion of hour", fontsize = 11)
    ax2.set_ylim([0, 0.85])

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc = "upper right", fontsize = 10)

    ax.set_title("Average diurnal profile (over all weeks and days)", fontsize = 12, fontweight = "bold")
    ax.grid(True, alpha = 0.3)
    ax.set_xticks(hours_of_day)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/diurnal_profile.png", dpi = 150, bbox_inches = "tight")
    print(f"Saved diurnal profile plot to: {output_dir}/diurnal_profile.png")
    plt.close()


def draw_outcome_distributions(df, output_dir):
    """Draws distribution histograms for all prediction target variables."""
    # Select target variables
    numeric_targets = NUMERICS
    binary_targets = BINARY

    n_targets = len(numeric_targets) + len(binary_targets)
    ncols = 3
    nrows = (n_targets + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize = (ncols * 4.7, nrows * 4), squeeze = False)
    axes_flat = list(axes.flat)

    axis_idx = 0

    for target in numeric_targets:
        ax = axes_flat[axis_idx]
        data = df[target].dropna()
        ax.hist(data, bins = 30, color = "#2196F3", edgecolor = "black", alpha = 0.7)

        mean_val = data.mean()
        ax.axvline(mean_val, color = "#F44336", linestyle = "--", linewidth = 2, label = f"mean={mean_val:.1f}")

        title_names = {
            "bdi": "BDI (depression)",
            "mmse": "MMSE (cognition)",
            "sf36": "SF-36 (HRQoL)",
            "chair_stand": "Chair Stand (mobility)",
            "sedentary": "Sedentary behaviour"
        }
        ax.set_title(title_names.get(target, target), fontweight = "bold", fontsize = 11)
        ax.set_ylabel("weeks", fontsize = 10)
        ax.legend(fontsize = 9, loc = "upper right")
        ax.grid(True, alpha = 0.3, axis = "y")
        axis_idx += 1

    for target in binary_targets:
        ax = axes_flat[axis_idx]
        data_binary = df[target].dropna()
        counts = data_binary.value_counts().sort_index()
        labels = ["No (0)", "Yes (1)"]
        colors = ["#2196F3", "#F44336"]

        bars = ax.bar(labels[:len(counts)], counts.values, color = colors[:len(counts)], 
                     edgecolor = "black", alpha = 0.7)

        for bar, count in zip(bars, counts.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f"{int(count)}",
                   ha = "center", va = "bottom", fontsize = 10, fontweight = "bold")

        title_names = {"DMT2": "Type-2 diabetes"}
        ax.set_title(title_names.get(target, target), fontweight = "bold", fontsize = 11)
        ax.set_ylabel("weeks", fontsize = 10)
        ax.grid(True, alpha = 0.3, axis = "y")
        axis_idx += 1

    for idx in range(axis_idx, len(axes_flat)):
        axes_flat[idx].axis("off")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/outcome_distributions.png", dpi = 150, bbox_inches = "tight")
    print(f"Saved outcome distributions plot to {output_dir}/outcome_distributions.png")
    plt.close()


def format_statistics_tables(descriptor_stats, covariates_stats):
    """Print formatted statistics tables to console."""
    print("\n" + "=" * 80)
    print("DESCRIPTORS STATISTICS")
    print("="*80)
    print(descriptor_stats.to_string(index=False))

    print("\n" + "=" * 80)
    print("COVARIATES AND TARGETS STATISTICS")
    print("=" * 80)
    print(covariates_stats.to_string(index=False))


def main():
    """Main function to run the EDA pipeline."""
    # Set working directory to the directory of this script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Create output directory
    output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok = True)

    print(f"Output directory: {output_dir}")
    print("\n" + "=" * 80)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 80)

    # Set seed for reproducibility
    set_seed()

    # Load data
    print("\nLoading data...")
    tensor, df = load_data()
    print(f"Tensor shape: {tensor.shape} (weeks, hours, descriptors)")
    print(f"DataFrame shape: {df.shape} (records, variables)")

    # Compute statistics
    print("\nComputing descriptor statistics...")
    descriptor_stats = compute_descriptors_statistics(tensor)

    print("Computing covariates and targets statistics...")
    covariates_stats = compute_covariates_targets_statistics(df)

    # Save tables as CSV
    print("\nSaving tables...")
    save_tables(descriptor_stats, covariates_stats, output_dir)

    # Print tables to console
    format_statistics_tables(descriptor_stats, covariates_stats)

    # Create visualizations
    print("\nCreating visualizations...")
    draw_correlation_heatmap(df, output_dir)
    draw_demographics(df, output_dir)
    draw_diurnal_profile(tensor, output_dir)
    draw_outcome_distributions(df, output_dir)

    print("\n" + "=" * 80)
    print("EDA COMPLETED")
    print("=" * 80)
    print(f"\nAll outputs saved to: '{output_dir}/'")


if __name__ == "__main__":
    main()
