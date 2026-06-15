from sys import meta_path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from config import BINARY, TARGET_COLS


class WeeklyAccelerometryDataset(Dataset):
    def __init__(self, descriptors, ClinicalCovariates, OutcomeTargets, mask, df, split, augment = False):
        # Indices of the rows belonging to the split selected
        self.indices = df.index[df["split"] == split].to_numpy()
        # Data tensor of the Descriptors values of the selected rows
        self.descriptors = torch.tensor(descriptors[self.indices], dtype = torch.float32)
        # Data tensor with the Clinical Covariates values of the selected rows
        self.ClinicalCovariates = torch.tensor(ClinicalCovariates[self.indices], dtype = torch.float32)
        # Data tensor with the values that we want to predict of the selected rows
        self.OutcomeTargets = torch.tensor(OutcomeTargets[self.indices], dtype = torch.float32)
        # Data tensor with the mask values for the selected rows
        self.mask = torch.tensor(mask[self.indices], dtype = torch.float32)
        # Identifier of the participant for each row in the selected split
        self.participant_id = df.loc[self.indices, "participant_id"].to_numpy()
        # Whether we want to apply data augmentation -> Only active in the training split
        self.augment = augment

    # Amount of samples in the dataset (number of rows in the split selected)
    def __len__(self):
        return len(self.indices)

    # Returns the sample at the index selected, applying data augmentation if specified
    def __getitem__(self, idx):
        # If there is data augmentation, we clone the tensor to avoid modifying the original data
        d = self.descriptors[idx].clone()

        # if augment == TRUE
        if self.augment:
            # Random noise so that the model doesn't see the same date for each epoch
            # The model won't overfit and will generalize better to unseen data
            d = d + torch.randn_like(d) * 0.05

            # Simulates variability between descriptors and participants
            # Multiplies the tensor by a random number between 0.9 and 1.1
            d = d * torch.empty(1).uniform_(0.9, 1.1)

            # Randomly selects which hours will be visible (10% of the hours will be zeroed out)
            # Forces the model to be robust to the data gaps that exist in the real tensor
            visible_hour = torch.rand(d.shape[0]) > 0.10
            d = d * visible_hour.unsqueeze(1)

        return {
            "descriptor": d, # the values of the descriptors
            "covariate": self.ClinicalCovariates[idx], # the values of the clinical covariates
            "target": self.OutcomeTargets[idx], # the values that we want to predict
            "mask": self.mask[idx], # indicates which targets are valid (1) and which are missing (0)
            "participant_id": self.participant_id[idx], # the identifier of the participant
        }


# Loads, partitions and normalizes the data
# Returns loaders and scalers
def load_data(
    descriptors_path = "../data/tensor_X.npy",
    df_path = "../data/dfParticipants.parquet",
    batch_size_train = 64,
    batch_size_eval = 128,
    random_state = 42,
):
    # Load the data
    descriptors = np.load(descriptors_path)
    df = pd.read_parquet(df_path)

    # Partition at participan level (Training = 70; Validation = 15; Testing = 15)
    participants = df["participant_id"].drop_duplicates()
    strat_col = BINARY[0] if BINARY else None # DMT2
    # Group by Participant and get the maximum value of the stratification variable (DMT2)
    # It maintains similar proportions of cases with and without DMT2 in train, validation, and test
    participant_strat = (df.groupby("participant_id")[strat_col].max().reindex(participants)
                         if strat_col is not None else None)

    train_p, temp_p = train_test_split(
        participants, 
        test_size = 0.30, # 70% in training; 30% in validation and testing
        random_state = random_state,
        stratify = participant_strat
    )
    val_p, test_p = train_test_split(
        temp_p,
        test_size = 0.50, # 30% in validation (15%) and testing (15%)
        random_state = random_state,
        stratify = participant_strat.loc[temp_p] if participant_strat is not None else None,
    )

    # Initially, all participants are marked as "trained"
    df["split"] = "train"
    # Some participants are moved to validation
    df.loc[df["participant_id"].isin(val_p), "split"] = "validation"
    # Some participants are moved to testing
    df.loc[df["participant_id"].isin(test_p), "split"] = "test"

    # Checks that no participant appears in more than one split
    verify_no_leakage(df)

    # Get the indices of the training samples
    train_idx = df.index[df["split"] == "train"].to_numpy()

    # Flatten the accelerometry tensor to NORMALISE it (mean/std of the training dataset)
    descriptors_train_flat = descriptors[train_idx].reshape(-1, descriptors.shape[-1])
    # Compute the mean of each descriptor
    descriptors_mean = descriptors_train_flat.mean(axis = 0)
    # Compute the standard deviation of each descriptor
    # If the standard deviation is 0, we replace it with 1 to avoid division by zero
    descriptors_std  = np.where(descriptors_train_flat.std(axis = 0) == 0, 1.0, descriptors_train_flat.std(axis = 0))
    # Descriptors normalized (NaN -> 0)
    descriptors_scaled = np.nan_to_num((descriptors - descriptors_mean) / descriptors_std, nan = 0.0)

    # NORMALISE clinical covariants (bmi, age, sex)
    cov_cols  = [c for c in df.columns
                 if c not in ["idweek", "participant_id", "Tiempo", "split"] + TARGET_COLS]
    # Get the clinical covariates of the training samples
    cov_train = df.loc[train_idx, cov_cols]
    # Compute the mean and standard deviation of each clinical covariate in the training set
    cov_mean  = cov_train.mean()
    # Compute the standard deviation of each clinical covariate in the training set (0 -> 1 to avoid division by zero)
    cov_std   = cov_train.std().replace(0, 1)
    # Normalise the clinical covariates of all samples (NaN -> 0)
    cov = np.nan_to_num(
        ((df[cov_cols] - cov_mean) / cov_std).to_numpy(dtype = "float32"), nan = 0.0
    )

    # MASKS and NORMALISATION of predictive outcomes
    target_raw = df[TARGET_COLS].to_numpy(dtype = "float32") # DMT2, sf36, sedentary, bdi, mmse, chair_stand
    # Whether the targets are valid (1) or missing (0)
    mask = (~df[TARGET_COLS].isna()).to_numpy(dtype = "float32")
    # Normalise the numeric targets using the mean and standard deviation of the training set
    target_mean, target_std, numeric_targets_standardised = normalize_numeric_targets(target_raw, mask, train_idx)

    # Build DataLoaders
    # Augment = True just in train split
    make_ds = lambda split, augment = False: WeeklyAccelerometryDataset(
        descriptors_scaled,
        cov,
        numeric_targets_standardised,
        mask,
        df,
        split,
        augment
    )
    # Create DataLoaders for training
    train_loader = DataLoader(
        make_ds("train", augment = True),
        batch_size = batch_size_train,
        shuffle = True
    )
    # Create DataLoaders for validation
    val_loader = DataLoader(
        make_ds("validation"),
        batch_size = batch_size_eval,
        shuffle = False
    )
    # Create DataLoaders for testing
    test_loader = DataLoader(
        make_ds("test"),
        batch_size = batch_size_eval,
        shuffle = False
    )

    return {
        "train_loader": train_loader, # DataLoader for training
        "val_loader": val_loader, # DataLoader for validation
        "test_loader": test_loader, # DataLoader for testing
        "d": descriptors_scaled.shape[-1], # Number of descriptors
        "seq_len": descriptors_scaled.shape[1], # Number of hours
        "cov_dim": cov.shape[-1], # Dimensionality of clinical covariates
        "target_mean": target_mean, # Mean of the targets in the training set
        "target_std": target_std, # Standard deviation of the targets in the training set
    }


# Checks that no participant appears in more than one split
def verify_no_leakage(df):
    # Get the set of participants in each split
    splits = {s: set(g["participant_id"]) for s, g in df.groupby("split")}
    # Check that the participants in "train" are not in "validation"
    assert splits["train"].isdisjoint(splits["validation"])
    # Check that the participants in "train" are not in "test"
    assert splits["train"].isdisjoint(splits["test"])
    # Check that the participants in "validation" are not in "test"
    assert splits["validation"].isdisjoint(splits["test"])


# Standarizes the numeric targets using statistics from the training set. Binary targets are not scaled.
def normalize_numeric_targets(target_raw, mask, train_idx):
    # Mean array of the targets initally set to 0
    target_mean = np.zeros(len(TARGET_COLS), dtype = "float32")
    # Standard deviation array of the targets initally set to 1
    target_std  = np.ones(len(TARGET_COLS),  dtype = "float32")
    for i, col in enumerate(TARGET_COLS):
        # We don't want to standardize the binary targets
        if col in BINARY:
            continue
        # Get the valid values of the target in the training set (mask == 1)
        valid_target = target_raw[train_idx, i][mask[train_idx, i] == 1]
        if len(valid_target) > 0:
            # Mean of the target's valid values
            target_mean[i] = valid_target.mean()
            # Standard deviation of the target's valid values
            target_std[i]  = valid_target.std() if valid_target.std() > 0 else 1.0
    # Create a copy of the target values to avoid modifying the original data
    targets = target_raw.copy()
    # Standardize the numeric targets using the mean and standard deviation of the training set
    for i, col in enumerate(TARGET_COLS):
        if col not in BINARY:
            targets[:, i] = (target_raw[:, i] - target_mean[i]) / target_std[i]
    # return the mean and standard deviation of the targets, and the standardized targets with NaN replaced by 0
    return target_mean, target_std, np.nan_to_num(targets, nan = 0.0).astype("float32")
