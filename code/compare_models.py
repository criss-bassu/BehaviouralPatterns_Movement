import itertools
import os
import math
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from config import (
    BINARY,
    NUMERICS,
    FINAL_TRAINING,
    GRID_TRAINING,
    MODEL_SPECS,
    OUTPUT_DIR,
    SEED,
    TARGET_COLS,
)
from preprocessing_01 import load_data
from models_02 import MLPEncoder, CNNEncoder, GRUEncoder, WeeklyOutcomeModel
from training_03 import fit_model
from evaluation_04 import collect_predictions, participant_bootstrap

# Creates a dictionary that assigns the main metric to each task based on its type
# Main metric per task: AUC-ROC for binary tasks, RMSE for numeric tasks
# ** Unpacks the dictionary -> We merge both into one
METRIC_NAME = {**{t: "auc_roc" for t in BINARY},
               **{t: "rmse"    for t in NUMERICS}}

REGRESSION_TASKS = NUMERICS

def set_seed(seed = SEED):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def grid(n, max_cols):
    """Rows and columns needed for a grid of n subplots."""
    # Calculate the number of columns, ensuring it does not exceed max_cols
    ncols = min(max_cols, n)
    # Calculate the number of rows needed to accommodate all subplots
    nrows = math.ceil(n / ncols) #.ceil = Rounds up to the nearest integer
    return nrows, ncols


def build_encoder(model_name, data, hp):
    """Builds an encoder based on the specified name and hyperparameters."""
    if model_name == "MLP":
        return MLPEncoder(
            d = data["d"],
            hours = data["hours"],
            hidden_dim = hp["hidden_dim"],
            rep_dim = hp["rep_dim"],
            dropout = hp["dropout"],
        )
    if model_name == "CNN":
        return CNNEncoder(
            d = data["d"],
            channels = hp["channels"],
            rep_dim = hp["rep_dim"],
            dropout = hp["dropout"],
            kernel_size = hp["kernel_size"],
        )
    if model_name == "GRU":
        return GRUEncoder(
            d = data["d"],
            hidden_dim = hp["hidden_dim"],
            rep_dim = hp["rep_dim"],
            num_layers = hp["num_layers"],
            dropout = hp["dropout"],
        )
    raise ValueError(f"Model {model_name} not recognized")


def build_model(model_name, data, hp):
    """Builds a WeeklyOutcomeModel with the specified encoder and hyperparameters."""
    encoder = build_encoder(model_name, data, hp)
    return WeeklyOutcomeModel(
        encoder = encoder,
        tasks = TARGET_COLS,
        rep_dim = hp["rep_dim"],
        cov_dim = data["cov_dim"],
        head_hidden = hp["head_hidden"],
        head_dropout = hp["head_dropout"]
    )


def fit_with_hp(model_name, data, device, hp, training):
    """Builds and trains a model with the given hyperparameters."""
    model = build_model(model_name, data, hp)
    # TRAIN the MODEL, and return the training history and the best model
    model, history_df = fit_model(
        model = model, # WeeklyOutcomeModel with the encoder and the tasks to predict
        train_data = data["train_data"], # training data
        val_data = data["val_data"], # validation data
        device = device,
        lr = hp["lr"],  # learning rate
        weight_decay = hp["weight_decay"], # L2 regularization -> prevents overfitting by penalizing large weights
        max_epochs = training["max_epochs"], # maximum number of epochs to train the model
        patience = training["patience"], # Early stopping -> Prevents overfitting when the validation loss stops improving for a number of epochs
        warmup_epochs = training["warmup_epochs"],
    )
    return model, history_df


def coerce_grid_value(value, grid_values):
    """Coerces the value to the type of the first element in grid_values."""
    template = grid_values[0]
    if isinstance(template, int) and not isinstance(template, bool):
        return int(value)
    if isinstance(template, float):
        return float(value)
    return value


def grid_search_model(name, spec, datos, device, output_dir):
    """"Performs a grid search for the specified model and hyperparameter grid."""
    grid = spec["grid"]
    keys = list(grid.keys())
    combinations = list(itertools.product(*grid.values()))

    print(f"\n{'=' * 55}\n  Grid search: {name} ({len(combinations)} combinations)\n{'=' * 55}")

    rows = []
    for i, values in enumerate(combinations, 1):
        candidate = dict(zip(keys, values))
        hp = {**spec["base_hp"], **candidate}
        set_seed()

        _, history_df = fit_with_hp(name, datos, device, hp, GRID_TRAINING)
        best_row = history_df.loc[history_df["val_loss"].idxmin()]
        val_loss = float(best_row["val_loss"])

        rows.append({
            **candidate,
            "val_loss": val_loss,
            "best_epoch": int(best_row["epoch"]),
            "epochs_run": int(len(history_df)),
        })
        print(f"  [{i:2d}/{len(combinations)}] {candidate} -> val_loss = {val_loss:.4f}")

    table = pd.DataFrame(rows).sort_values("val_loss").reset_index(drop=True)
    table.to_csv(f"{output_dir}/grid_search_{name}.csv", index=False)
    print(f"\n=== Best configuration {name} ===")
    print(table.head(5).to_string(index=False))

    best_values = {key: coerce_grid_value(table.iloc[0][key], grid[key]) for key in keys}
    best_hp = {**spec["base_hp"], **best_values}
    return best_hp, table


def run_model(model_name, hp, data, device):
    """Trains and evaluates a model with the specified encoder and hyperparameters."""
    print(f"\n{'=' * 55}\n  Final Training: {model_name}\n{'=' * 55}")
    print("  Hyperparameters:", {k: hp[k] for k in sorted(hp)})

    set_seed()
    model, history_df = fit_with_hp(model_name, data, device, hp, FINAL_TRAINING)
    print(f"  Epochs: {len(history_df)}  |  Best validation loss: {history_df['val_loss'].min():.4f}")

    # EVALUATE the MODEL on the TEST set -> Get the predictions of the model over the test set
    test_preds = collect_predictions(model, data["test_data"], device = device)

    results = {} # Initialize a dictionary to store the results for each task
    for task in TARGET_COLS:
        metric = METRIC_NAME[task] # auc-roc for binary targets; rmse for numeric targets
        # Handling errors to ensure that the evaluation continues even if one task fails
        try:
            # For each target variable, compute the main metric and its 95% confidence interval using bootstrapping
            boot = participant_bootstrap(
                test_preds, # DataFrame with predictions, real values and masks for each task
                task, # The target variable to evaluate
                metric, # The main metric to compute for the target variable
                target_mean = data["target_mean"], # Mean of the target variable in the training set (used for scaling back to original units)
                target_std = data["target_std"], # Standard deviation of the target variable in the training set (used for scaling back to original units)
                n_boot = 1000, # Number of bootstrap iterations to compute the confidence interval
                seed = 42, # Random seed for reproducibility of the bootstrap results
            )
            # Store values of the metric for the current task in the dictionary
            results[task] = {
                "mean": boot["mean"], # Mean of the metric computed over the bootstrap samples
                 "ic_2_5": boot["lower_2_5"], # Lower bound of the 95% confidence interval (2.5th percentile)
                 "ic_97_5": boot["upper_97_5"] # Upper bound of the 95% confidence interval (97.5th percentile)
            }
        except Exception:
            results[task] = {
                "mean": float("nan"),
                "ic_2_5": float("nan"),
                "ic_97_5": float("nan")
            }

        if task in REGRESSION_TASKS:
            try:
                boot_r = participant_bootstrap(
                    test_preds,
                    task,
                    "pearson_r",
                    target_mean = data["target_mean"],
                    target_std = data["target_std"],
                    n_boot = 1000,
                    seed = 42,
                )
                results[task]["pearson_r"] = boot_r["mean"]
                results[task]["pearson_r_ic_2_5"] = boot_r["lower_2_5"]
                results[task]["pearson_r_ic_97_5"] = boot_r["upper_97_5"]
            except Exception:
                results[task]["pearson_r"] = float("nan")
                results[task]["pearson_r_ic_2_5"] = float("nan")
                results[task]["pearson_r_ic_97_5"] = float("nan")

    # Return the training history and the evaluation results for each target variable
    return history_df, results


def save_best_hyperparameters(best_hps, output_dir):
    rows = []
    for model_name, hp in best_hps.items():
        row = {"Model": model_name}
        row.update(hp)
        rows.append(row)
    tabla = pd.DataFrame(rows)
    tabla.to_csv(f"{output_dir}/best_hyperparameters.csv", index = False)
    print("\n=== BEST HYPERPARAMETERS ===")
    print(tabla.to_string(index = False))


def save_table(all_results, output_dir):
    """Saves a CSV file with the evaluation results for each model and task"""
    rows = []
    # Iterate through each model and its corresponding results
    for model_name, results in all_results.items():
        # Iterate through the targets
        for task in TARGET_COLS:
            res = results[task]
            row = {
                "model": model_name,
                "task": task,
                "metric": METRIC_NAME[task],
                "mean": round(res["mean"], 4), # Round the mean of the metric to 4 decimal places
                "ic 2.5%": round(res["ic_2_5"], 4),
                "ic 97.5%": round(res["ic_97_5"], 4),
                # If the task is a regression task, include Pearson
                "pearson_r": round(res["pearson_r"], 4) if task in REGRESSION_TASKS else "",
                "pearson_r IC 2.5%": round(res["pearson_r_ic_2_5"], 4) if task in REGRESSION_TASKS else "",
                "pearson_r IC 97.5%":round(res["pearson_r_ic_97_5"],4) if task in REGRESSION_TASKS else ""
            }
            rows.append(row)
    table = pd.DataFrame(rows)
    # index = False -> Do not include a column showing the index
    table.to_csv(f"{output_dir}/models_comparison.csv", index = False)
    print("\n=== COMPARATIVE TABLE ===")
    print(table.to_string(index = False))


def draw_loss_curves(histories, output_dir):
    """Draws the training and validation loss curves for each model"""
    # Creates a figure with 1 row and 3 columns of subplots, each subplot will show the loss curves for one model
    fig, axes = plt.subplots(1, 3, figsize = (15, 4))
    # Iterate through each axis and each model's corresponding training history
    for ax, (model_name, df) in zip(axes, histories.items()):
        # Draw the training loss curves for the current model
        ax.plot(df["epoch"], df["train_loss"], color = "#2196F3", lw = 2, label = "Train")
        # Draw the validation loss curves for the current model
        ax.plot(df["epoch"], df["val_loss"], color = "#F44336", lw = 2, label = "Val")
        # Highlight the epoch with the best validation loss (lowest value)
        best = df.loc[df["val_loss"].idxmin()]
        # Draw a vertical dashed line at the epoch with the best validation loss and mark it with a red dot
        ax.axvline(best["epoch"], color = "gray", linestyle = "--", lw = 1, alpha = 0.7)
        ax.scatter([best["epoch"]], [best["val_loss"]], color = "#F44336", zorder = 5, s = 50)
        # Set the title, labels, legend, and grid for the current subplot
        ax.set_title(f"{model_name} (epoch {int(best['epoch'])}, val = {best['val_loss']:.2f})", fontsize = 11, fontweight = "bold")
        ax.set_xlabel("Epoch") # Set the x-axis label to "Epoch"
        ax.set_ylabel("Loss") # Set the y-axis label to "Loss"
        ax.legend(fontsize = 9) # Set the legend font size to 9
        ax.grid(True, alpha = 0.3) # Set the grid with a transparency of 0.3
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer = True)) # Set the x-axis epochs to be integers only
    fig.suptitle("Curves of loss by architecture", fontsize = 13, fontweight = "bold") # Set the main title of the figure
    plt.tight_layout() # Adjust the layout of the subplots to prevent overlapping
    # dpi= 150 -> resolution of the saved image (higher dpi = better quality)
    plt.savefig(f"{output_dir}/loss_curves_comparison.png", dpi = 150)


def draw_metrics_comparison(all_results, output_dir):
    """Draws a bar plot comparing the main metric for each model and task"""
    # Define a color for each model
    model_colors = {"MLP": "#FF9800", "CNN": "#2196F3", "GRU": "#4CAF50"}
    # Calculate the number of rows and columns needed for the grid of subplots, with a maximum of 3 columns
    nrows, ncols = grid(len(TARGET_COLS), max_cols = 3)
    # Create a figure with the calculated number of rows and columns of subplots
    fig, axes = plt.subplots(nrows, ncols, figsize = (ncols * 4.7, nrows * 4), squeeze = False)
    axes_flat = list(axes.flat) # Flatten the 2D array of axes into a 1D list for easier iteration
    # Iterate through each axis and each target variable to plot the metrics
    for ax, task in zip(axes_flat, TARGET_COLS):
        # Get the main metric for the current target variable (AUC-ROC for binary tasks, RMSE for numeric tasks)
        metric = METRIC_NAME[task] 
        # Get the mean of the metric for each model
        means = [all_results[m][task]["mean"]  for m in all_results]
        # Get the lower bound of the 95% confidence interval for each model
        lowers = [all_results[m][task]["mean"] - all_results[m][task]["ic_2_5"]  for m in all_results]
        # Get the upper bound of the 95% confidence interval for each model
        uppers = [all_results[m][task]["ic_97_5"] - all_results[m][task]["mean"] for m in all_results]

        # Draw a bar plot for the current target variable
        bars = ax.bar(
            list(all_results.keys()), # x-axis labels = model names
            means, # y-axis values = mean of the metric for each model
            color = [model_colors[m] for m in all_results], # colors for each model
            yerr = [lowers, uppers], # error bars = lower and upper bounds of the 95% confidence interval
            capsize = 6, # size of the caps on the error bars
            error_kw = {"lw": 1.5} # line width of the error bars
        )
        # Set the titlefor the current subplot
        ax.set_title(f"{task} ({metric.upper()})", fontweight = "bold")
        # Set the y-axis label for the current subplot
        ax.set_ylabel(metric.upper())
        # Set the y-axis limits based on the type of task (0-1 for binary tasks, auto for numeric tasks)
        ax.grid(True, axis = "y", alpha = 0.3)

        # Highlight the best model for the current target variable based on the main metric
            # For binary tasks, the best model is the one with the highest AUC-ROC
            # For numeric tasks, the best model is the one with the lowest RMSE
        best_idx = means.index(max(means) if metric == "auc_roc" else min(means))
        bars[best_idx].set_edgecolor("black") # Set the edge color of the best model's bar to black
        bars[best_idx].set_linewidth(2.5) # Set the line width of the best model's bar to 2.5

    # Turn off the axes for any unused subplots
    # If the number of target variables is less than the total number of subplots
    for ax in axes_flat[len(TARGET_COLS):]:
        ax.axis("off")

    # Set the main title for the entire figure
    fig.suptitle("Metric comparison in test (bootstrap 1000 it.)\n"
                 "The best model per task is highlighted in bold",
                 fontsize = 12, # Set the font size of the title to 12
                 fontweight = "bold" # Set the font weight of the title to bold
                 )
    plt.tight_layout() # Adjust the layout of the subplots to prevent overlapping
    plt.savefig(f"{output_dir}/metrics_comparison.png", dpi = 150)


def draw_pearson(all_results, output_dir):
    """Draws a bar plot comparing the Pearson correlation for each model and regression task"""
    model_colors = {"MLP": "#FF9800", "CNN": "#2196F3", "GRU": "#4CAF50"}
    nrows, ncols = grid(len(REGRESSION_TASKS), max_cols = 5)
    fig, axes = plt.subplots(nrows, ncols, figsize = (ncols * 3.2, nrows * 4), squeeze = False)
    axes_flat = list(axes.flat)
    for ax, task in zip(axes_flat, REGRESSION_TASKS):
        means = [all_results[m][task]["pearson_r"] for m in all_results]
        lowers = [all_results[m][task]["pearson_r"] - all_results[m][task]["pearson_r_ic_2_5"] for m in all_results]
        uppers = [all_results[m][task]["pearson_r_ic_97_5"] - all_results[m][task]["pearson_r"] for m in all_results]

        bars = ax.bar(list(all_results.keys()), 
                      means,
                      color = [model_colors[m] for m in all_results],
                      yerr = [lowers, uppers], 
                      capsize = 6, 
                      error_kw = {"lw": 1.5})
        ax.set_title(task, fontweight = "bold")
        ax.set_ylabel("Pearson r")
        ax.set_ylim(-1, 1)
        ax.axhline(0, color = "gray", lw = 0.8, linestyle = "--")
        ax.grid(True, axis = "y", alpha = 0.3)

        valid = [v for v in means if not (v != v)]  # Exclude NaN values
        if valid:
            best_idx = means.index(max(valid))
            bars[best_idx].set_edgecolor("black")
            bars[best_idx].set_linewidth(2.5)

    for ax in axes_flat[len(REGRESSION_TASKS):]:
        ax.axis("off")

    fig.suptitle("Pearson Correlation — Numerical Health Results\n"
                 "(bootstrap 1000 it., IC 95%)  |  Black Border = Best Model",
                 fontsize = 12, 
                 fontweight = "bold")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pearson_correlation.png", dpi = 150)


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = OUTPUT_DIR + "/comparison"
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok = True) # exist_ok = True -> If the directory already exists, do not raise an error
    print(f"Device: {device}")

    data = load_data()

    best_hps = {}
    for name, spec in MODEL_SPECS.items():
        if spec.get("use_grid_search", True):
            best_hps[name], _ = grid_search_model(name, spec, data, device, output_dir)
        else:
            best_hps[name] = dict(spec["base_hp"])
            print(f"\n{name}: using fixed hyperparameters, without grid search")

    save_best_hyperparameters(best_hps, output_dir)

    histories = {} # Initialize a dictionary to store the evolution of the losses for each model
    all_results = {} # Initialize a dictionary to store the results for each model
    # Iterate through each model and its corresponding configuration
    for model_name, hp_cfg in best_hps.items():
        # Train and evaluate the model, and store the training history and evaluation results
        histories[model_name], all_results[model_name] = run_model(model_name, hp_cfg, data, device)

    save_table(all_results, output_dir) # Save a CSV file with the evaluation results for each model and task
    draw_loss_curves(histories, output_dir) # Draw the training and validation loss curves for each model
    draw_metrics_comparison(all_results, output_dir) # Draw a bar plot comparing the main metric for each model and task
    draw_pearson(all_results, output_dir) # Draw a bar plot comparing the Pearson correlation for each model and regression task
    print(f"\nResults in: '{output_dir}/'") # Print the directory where the results are saved
    print("=== COMPARISON COMPLETED ===")


if __name__ == "__main__":
    main()
