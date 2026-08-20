import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                              confusion_matrix, mean_absolute_error,
                              root_mean_squared_error, r2_score)
from scipy.stats import pearsonr

from config import BINARY, TARGET_COLS, TARGET_INDEX


# threshold = < 0.5 = 0; >= 0.5 = 1
def binary_classification_metrics(real_target, logits, threshold = 0.5):
    """Returns binary classification metrics based on real target and logits (predictions without sigmoid)."""
    # Apply the sigmoid function to convert logits to probabilities
    prob = 1 / (1 + np.exp(-logits))
    # Converts probabilities to binary predictions
    # .astype(int) converts TRUE/FALSE to 1/0)
    pred = (prob >= threshold).astype(int) # IF prob < 0.5 Then 0; IF prob >= 0.5 THEN 1
    # Compute the confusion matrix to get True Negatives, False Positives, False Negatives, and True Positives
    # [[TN, FP],
    #  [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(real_target, pred, labels = [0, 1]).ravel() # .ravel() flattens the matrix into an array
    return {
        # AUC-ROC measures how good the model classifies positive and negative samples
        # Needs probibilities, not binary predictions
        # len(np.unique(real_target)) == 2 checks if both classes exist for the target to predict
        "auc_roc": roc_auc_score(real_target, prob) if len(np.unique(real_target)) == 2 else np.nan,
        # Accuracy: Number of correct predictions / Total number of predictions
        "accuracy": accuracy_score(real_target, pred),
        # F1 score is the harmonic mean of precision and recall -> for unabalaced datasets
        # zero_division = 0 -> If there are no positive predictions, return 0 instead of raising an error
        "f1": f1_score(real_target, pred, zero_division = 0),
        # True Positive Rate: True positives / (True positives + False negatives)
        # Of all the real positive cases, how many did we correctly predict as positive?
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
        # True Negative Rate: True negatives / (True negatives + False positives)
        # Of all the real negative cases, how many did we correctly predict as negative?
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan
    }


def regression_metrics(real_target, pred_target):
    """Returns regression metrics based on real and predicted targets."""
    # The Pearson correlation coefficient measures the linear correlation between two variables
        # r = 0 -> no correlation
        # r = 1 -> perfect positive correlation
        # r = -1 -> perfect negative correlation
    r = (pearsonr(real_target, pred_target)[0] # Returns the correlation coefficient (r) and p-value, but we only need r
         # There must be at least 2 unique values in the target to predict
         # The value of the real and predictive target can't be constant
         # Otherwise the correlation is undefined
         if len(real_target) > 1 and np.std(real_target) > 0 and np.std(pred_target) > 0
         else np.nan)
    return {
        # MAE: measures the Mean Absolute Error between the real and predicted values: mean(|real - pred|)
        # In the same units as the target variable
        "mae": mean_absolute_error(real_target, pred_target),
        # RMSE: measures the Root Mean Squared Error between the real and predicted values: sqrt(mean((real - pred)^2))
        # Penalizes larger errors more heavily and prevents positive and negative errors from cancelling each other out
        "rmse": root_mean_squared_error(real_target, pred_target),   
        # Pearson correlation coefficient
        "pearson_r": r,
        # Coefficient of determination: measures how well the model explains the variance in the real target variable
            # R^2 = 1 -> perfect prediction
            # R^2 = 0 -> model predicts the mean of the target variable
            # R^2 < 0 -> model performs worse than predicting the mean of the target
        "r2": r2_score(real_target, pred_target)
    }


@torch.no_grad()
def collect_predictions(model, test_data, device):
    """Returns a DataFrame with predictions, real values and masks, for each task."""
    model.eval()
    rows = [] # Initialize a list to store the rows of the DataFrame
    for batch in test_data:
        d = batch["descriptor"].to(device)
        cov = batch["covariate"].to(device)
        outputs = model(d, cov) # Forward pass: model predicts the outcomes based on the input data

        # Convert the outputs to numpy arrays for easier manipulation (for each target to predict)
        out_np = {task: outputs[task].cpu().numpy() for task in TARGET_COLS}
        # Convert the real target values and masks to numpy arrays for easier manipulation
        target_np = batch["target"].numpy()
        mask_np = batch["mask"].numpy()

        # Iterate throught the participant's of the bacth
        for i, idx in enumerate(batch["participant_id"]):
            # Create a dictionary for each participant
            row = {"participant_id": idx}
            for task in TARGET_COLS:
                j = TARGET_INDEX[task] # Get the column index for the current target
                row[f"{task}_real"] = target_np[i, j] # Get the real value of the current target for the current participant
                row[f"{task}_pred"] = out_np[task][i] # Get the predicted value of the current target for the current participant
                row[f"{task}_mask"] = mask_np[i, j] # Get the mask value of the current target for the current participant
            rows.append(row)

    return pd.DataFrame(rows)


def compute_task_metric(df, task, metric_name, target_mean, target_std):
    """Compute a metric for a task over the rows with mask = 1, in original scale."""
    # Only weeks where the target has values
    valid = df[df[f"{task}_mask"] == 1]
    real_target = valid[f"{task}_real"].to_numpy()
    pred_target = valid[f"{task}_pred"].to_numpy()

    # If the target task is binary, compute the binary classification metrics:
    if task in BINARY:
        return binary_classification_metrics(real_target.astype(int), pred_target)[metric_name]
        # real_target.astype(int) converts the real target values to integers (0 or 1)
        # pred_target is the predicted logits (no probabilities)
        # [metric_name] returns the metric we're interested in

    # If the target task is not binary, compute the regression metrics:
    j = TARGET_INDEX[task]
    # De-normalization: rescale the real and predicted target values to their original scale
    real_target = real_target * target_std[j] + target_mean[j]
    pred_target = pred_target * target_std[j] + target_mean[j]
    return regression_metrics(real_target, pred_target)[metric_name]


def participant_bootstrap(test_preds, task, metric_name, target_mean, target_std, n_boot = 1000, seed = 9626):
    """Bootstrap by participant. Returns mean and 95% CI."""
    # Use a random number generator with a fixed seed for reproducibility
    rng = np.random.default_rng(seed)
    # Group the DataFrame by participant_id to create a dictionary
        # keys = participant IDs
        # values = DataFrames with the rows for that participant
    groups = {idx: grp for idx, grp in test_preds.groupby("participant_id")}
    # List of unique participant IDs
    participants = np.array(list(groups.keys()))

    values = [] # Initialize a list to store the metric values for each bootstrap sample
    for _ in range(n_boot):
        # Sample participants with replacement to create a bootstrap sample
        # size = len(participants) -> The size of the sample is equal to the number of participants
        # replace = True -> A participant can show up in between 0-times and several-times in the same sample
        sampled = rng.choice(participants, size = len(participants), replace = True)
        # For each participant selected in sampled, get their rows and concatenate them into a single DataFrame
        # ignore_index = True -> Restarts the index of the new DataFrame from 0 (preseves the original structure by participant)
        boot_df = pd.concat([groups[idx] for idx in sampled], ignore_index = True)
        # Catches possible errors
        try:
            # Compute the metric for the current bootstrap sample and append it to the values list
            values.append(compute_task_metric(boot_df, task, metric_name, target_mean, target_std))
        except ValueError:
            values.append(np.nan)

    values = np.array(values, dtype = float)
    # Keep only the non-NaN values
    values = values[~np.isnan(values)] # ~ inverts the boolean array
    return {
        "mean": np.mean(values), # mean of the metric values across all bootstrap samples
        "lower_2_5": np.percentile(values, 2.5), # lower bound of the 95% confidence interval (2.5th percentile)
        "upper_97_5": np.percentile(values, 97.5) # upper bound of the 95% confidence interval (97.5th percentile)
    }
