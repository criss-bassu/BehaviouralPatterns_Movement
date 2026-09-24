import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import BINARY, COVARIABLES, NUMERICS, OUTPUT_DIR_EDA, PARQUET_PATH, TENSOR_PATH, SEED


def get_descriptor_names():
    """Returns the names of all the accelerometry descriptors."""
    return [
        "ENMO_mean", "ENMO_sd", "ENMO_p50", "ENMO_p85", "ENMO_p90", "ENMO_p95",
        "SIB", "bed", "sleep", "steps", "ENMO050", "ENMO100", "ENMO200", "ENMO300"
    ]


def analyze_data_completeness(df):
    """Analyzes which weeks have incomplete variable data (non-descriptor variables).
    Returns a detailed report on missing data."""
    
    # Define all variable columns (excluding idweek, participant_id, Time)
    variable_columns = BINARY + NUMERICS + COVARIABLES
    
    print("\n\n--- DATA COMPLETENESS ANALYSIS ---\n")
    
    df['has_missing'] = df[variable_columns].isna().any(axis = 1)
    
    n_weeks_incomplete = df['has_missing'].sum()
    n_weeks_complete = (~df['has_missing']).sum()
    total_weeks = len(df)
    
    print(f"Total weeks in dataset: {total_weeks}")
    print(f"Weeks with ALL variables: {n_weeks_complete}")
    print(f"Weeks MISSING any variable: {n_weeks_incomplete}")
    
    # Clean up temporary column
    df.drop('has_missing', axis = 1, inplace = True)


def analyze_dmt2_distribution(df):
    """Analyzes DMT2 (Type 2 Diabetes) distribution"""
    
    print("\n\n--- DMT2 (TYPE 2 DIABETES) DISTRIBUTION ANALYSIS ---\n")
    
    dmt2_counts = df['DMT2'].value_counts()
    dmt2_missing = df['DMT2'].isna().sum()
    
    print(f"Value Counts:")
    print(f"\t 0 (No diabetes): {dmt2_counts.get(0.0, 0)}")
    print(f"\t 1 (Has diabetes): {dmt2_counts.get(1.0, 0)}")
    if dmt2_missing > 0:
        print(f"\t NaN (Missing): {dmt2_missing}")


def analyze_accelerometry_missing_values(tensor):
    """ Analyzes missing values in the accelerometry tensor.
    - How many missing values in total out of how many total values
    - How many weeks are affected by missing values """

    print("\n\n--- ACCELEROMETRY TENSOR: MISSING VALUES ANALYSIS ---\n")
    
    # Calculate total values
    total_values = tensor.size
    
    # Count missing values (NaN)
    missing_values = np.isnan(tensor).sum()
    
    # Find weeks with missing values
    # Check if any value in each week is NaN
    n_weeks_affected = 0
    total_weeks = tensor.shape[0]
    for week_idx in range(total_weeks):
        if np.isnan(tensor[week_idx]).any():
            n_weeks_affected += 1
    
    # Print results
    print(f"Total missing values: {missing_values:,} out of {total_values:,}")
    print(f"Weeks with missing data: {n_weeks_affected} out of {total_weeks}")


def analyze_steps_outliers(df):
    """Outlier analysis for the 'steps' descriptor."""
    
    steps_data = df[:, :, 9]  # Assuming the 'steps' descriptor is at index 9 in the tensor
    
    outlier_mask = (steps_data < 0) | (steps_data > 15000)
    
    n_outliers = np.sum(outlier_mask)
    n_weeks = np.sum(np.any(outlier_mask, axis = 1))
    
    print(f"\nOutliers in 'steps': {n_outliers:,} hours | {n_weeks} weeks afected")


def compute_descriptors_statistics(df):
    """Gets summary statistics for the hourly descriptors: Mean, SD, Min, Max."""
    descriptor_names = get_descriptor_names()

    # Flatten the tensor
    # df.shape[-1] = number of descriptors
    df_flat = df.reshape(-1, df.shape[-1])

    stats = []
    for i, desc_name in enumerate(descriptor_names):
        data = df_flat[:, i] # Select the descriptor column
        # Remove NaN values
        data_clean = data[~np.isnan(data)]

        stats.append({
            "#": i + 1, # Index starting from 1
            "Descriptor": desc_name,
            "Mean": np.mean(data_clean),
            "SD": np.std(data_clean),
            "Min": np.min(data_clean),
            "Max": np.max(data_clean)
        })

    return pd.DataFrame(stats)


def compute_covariates_targets_statistics(df):
    """Get summary statistics for clinical covariates and prediction targets.
    Returns a DataFrame with columns: Variable, Type, n_valid, Missing, Mean, SD, Min, Max"""

    all_vars = BINARY + NUMERICS + COVARIABLES
    stats = []

    for var in all_vars:
        if var in COVARIABLES:
            var_type = "covariate"
        elif var in BINARY:
            var_type = "binary"
        else:
            var_type = "regression"

        data = df[var]
        stats.append({
            "Variable": var,
            "Type": var_type,
            "n_valid": data.notna().sum(),
            "Missing": data.isna().sum(),
            "Mean": data.mean(),
            "SD": data.std(),
            "Min": data.min(),
            "Max": data.max()
        })

    return pd.DataFrame(stats)


def save_tables(descriptor_stats, covariates_stats, output_dir):
    """Save statistics tables as CSV files."""
    # Save descriptor statistics
    desc_path = f"{output_dir}/table_descriptors_statistics.csv"
    descriptor_stats.to_csv(desc_path, index = False)
    print(f"\n\nSaved descriptors statistics table to: {desc_path}")

    # Save covariates and targets statistics
    cov_path = f"{output_dir}/table_covariates_targets_statistics.csv"
    covariates_stats.to_csv(cov_path, index = False)
    print(f"Saved covariates and targets statistics table to: {cov_path}")


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
        nrows = (n_plots + 2) // 3 # Determines the number of rows needed

    fig, axes = plt.subplots(nrows, ncols, figsize = (ncols * 4.7, nrows * 4))
    
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten() # One dimension array of axes

    axis_idx = 0
    ############################ NO COVARIABLES
    numeric_covariables = ["age", "bmi"]
    numeric_colors = {"age": "#9C27B0", "bmi": "#FF9800"}

    for cov in COVARIABLES:
        if cov in numeric_covariables:
            ax = axes[axis_idx]
            data = df_unique[cov].dropna()
            ax.hist(data, bins = 20, color = numeric_colors[cov], edgecolor = "black", alpha = 0.7)
            ax.set_title(f"{cov.upper()} (participants)", fontsize = 11, fontweight = "bold")
            ax.set_xlabel("years" if cov == "age" else "kg/m^2")
            ax.set_ylabel("count")
            ax.grid(True, alpha = 0.3)
            axis_idx += 1

        elif cov == "sex":
            ax = axes[axis_idx]
            data_sex = df_unique[cov].dropna()
            counts = data_sex.value_counts().sort_index()
            labels = ["Female (0)", "Male (1)"]
            colors = ["#FF6B9D", "#4A90E2"]

            bars = ax.bar(
                labels[:len(counts)],
                counts.values,
                color = colors[:len(counts)], 
                edgecolor = "black",
                alpha = 0.7
            )

            for bar, count in zip(bars, counts.values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height, f"{int(count)}",
                    ha = "center",
                    va = "bottom",
                    fontsize = 10,
                    fontweight = "bold"
                )

            ax.set_title(f"{cov.upper()} (participants)", fontsize = 11, fontweight = "bold")
            ax.set_ylabel("count")
            ax.grid(True, alpha = 0.3, axis = "y")
            axis_idx += 1
    ############################ NO COVARIABLES

    ax = axes[axis_idx]

    # Count how many participants provided each number of weeks
    week_counts = weeks_per_participant.value_counts().sort_index()

    bars = ax.bar(
        week_counts.index, # Number of weeks
        week_counts.values, # Number of participants
        color = "#2196F3",
        edgecolor = "black",
        alpha = 0.7
    )

    # Add the number of participants above each bar
    ax.bar_label(
        bars,
        labels = [str(count) for count in week_counts.values], # Convert counts to strings
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


def draw_diurnal_profile(df, output_dir):
    """Draws the average diurnal profile of accelerometry descriptors.
    Shows the mean activity pattern across all weeks and days, for ENMO_mean (activity level) and sleep proportion."""
    descriptor_names = get_descriptor_names()

    # Compute mean of descriptors across all weeks and participants (ignoring NaN values)
    mean_descriptors = np.nanmean(df, axis = 0)  # Shape: (168 hours, 14 descriptors)

    # Select indices for the descriptors of interest
    idx_enmo_mean = descriptor_names.index("ENMO_mean")
    idx_sleep = descriptor_names.index("sleep")

    # Compute hourly averages
    hours_of_day = np.arange(24)
    # Mean across each hour (168 hours / 7 days = 24 hours per day)
    enmo_hourly = np.array([mean_descriptors[h::24, idx_enmo_mean].mean() for h in range(24)])
    sleep_hourly = np.array([mean_descriptors[h::24, idx_sleep].mean() for h in range(24)])

    fig, ax = plt.subplots(figsize = (12, 5))

    # Primary axis: ENMO_mean (activity)
    color_enmo = "#2196F3"
    ax.plot(hours_of_day, enmo_hourly, marker = "o", color = color_enmo, linewidth = 2, label = "ENMO_mean (activity)")
    ax.set_xlabel("Hour of day", fontsize = 11)
    ax.set_ylabel("Mean ENMO (g)", fontsize = 11, color = color_enmo)
    ax.tick_params(axis = "y", labelcolor = color_enmo)

    # Secondary axis: Sleep proportion
    ax2 = ax.twinx()
    color_sleep = "#9C27B0"
    ax2.plot(hours_of_day, sleep_hourly, marker = "s", color = color_sleep, linewidth = 2, label = "Sleep proportion")
    ax2.set_ylabel("Proportion of hour", fontsize = 11)
    ax2.set_ylim([0, 1])

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels() # Primary axis
    lines2, labels2 = ax2.get_legend_handles_labels() # Secondary axis
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

    n_targets = len(NUMERICS) + len(BINARY)
    ncols = 3
    nrows = (n_targets + ncols - 1) // ncols # Determines the number of rows needed

    fig, axes = plt.subplots(nrows, ncols, figsize = (ncols * 4.7, nrows * 4), squeeze = False)
    axes_flat = list(axes.flat)

    axis_idx = 0

    for target in NUMERICS:
        ax = axes_flat[axis_idx]
        data = df[target].dropna()
        ax.hist(data, bins = 30, color = "#2196F3", edgecolor = "black", alpha = 0.7)

        mean_val = data.mean()
        ax.axvline(mean_val, color = "#F44336", linestyle = "--", linewidth = 2, label = f"mean = {mean_val:.1f}")

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

    for target in BINARY:
        ax = axes_flat[axis_idx]
        data_binary = df[target].dropna()
        counts = data_binary.value_counts().sort_index()
        labels = ["No (0)", "Yes (1)"]
        colors = ["#2196F3", "#F44336"]

        bars = ax.bar(
            labels[:len(counts)],
            counts.values,
            color = colors[:len(counts)], 
            edgecolor = "black",
            alpha = 0.7
        )

        # Set count labels above each bar
        for bar, count in zip(bars, counts.values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f"{int(count)}",
                ha = "center",
                va = "bottom",
                fontsize = 10,
                fontweight = "bold"
            )

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
    print("\n\n--- DESCRIPTORS STATISTICS ---\n")
    print(descriptor_stats.to_string(index = False))

    print("\n\n--- COVARIATES AND TARGETS STATISTICS ---\n")
    print(covariates_stats.to_string(index = False))


def main():
    """Main function to run the EDA pipeline."""
    # Set working directory to the directory of this script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Create output directory
    output_dir = OUTPUT_DIR_EDA
    os.makedirs(output_dir, exist_ok = True)

    print(f"Output directory: {output_dir}\n")

    # Set seed for reproducibility
    np.random.seed(SEED)

    # Load data
    tensor = np.load(TENSOR_PATH)  # Shape: (n_weeks, n_hours, n_descriptors)
    df = pd.read_parquet(PARQUET_PATH)  # Clinical covariates and targets

    print(f"Tensor shape: {tensor.shape} (weeks, hours, descriptors)")
    print(f"DataFrame shape: {df.shape} (records, variables)")

    # Analyze accelerometry missing values
    analyze_accelerometry_missing_values(tensor)

    # Analyze data completeness
    analyze_data_completeness(df)

    # Analyze DMT2 distribution
    analyze_dmt2_distribution(df)

    # Analyze outliers in 'steps' descriptor
    # analyze_steps_outliers(tensor)

    # Compute statistics
    descriptor_stats = compute_descriptors_statistics(tensor)

    covariates_stats = compute_covariates_targets_statistics(df)

    # Print tables to console
    format_statistics_tables(descriptor_stats, covariates_stats)

    # Save tables as CSV
    save_tables(descriptor_stats, covariates_stats, output_dir)

    # Create visualizations
    draw_correlation_heatmap(df, output_dir)
    draw_demographics(df, output_dir)
    draw_diurnal_profile(tensor, output_dir)
    draw_outcome_distributions(df, output_dir)

    print(f"\nAll outputs saved to: '{output_dir}/'")


if __name__ == "__main__":
    main()
