# Vectors of the variables to predict based on their type
# Classification tasks (BCE loss, metric AUC-ROC)
BINARY  = ["DMT2"] # Can be empty
# Regression tasks (MSE loss, metrics RMSE / Pearson)
NUMERICS = ["sf36", "sedentary", "bdi", "mmse", "chair_stand"] # can't be empty

# Order of the tasks: binary tasks first, then numeric tasks
TARGET_COLS = BINARY + NUMERICS
TARGET_INDEX  = {task: i for i, task in enumerate(TARGET_COLS)}


# COVARIABLES = ["sex", "age", "bmi"]
COVARIABLES = []


# Accelerometry tensor file path (Participants, hours, variables)
TENSOR_PATH = "../data/tensor_X-1.npy"
PARQUET_PATH = "../data/dfParticipants-1.parquet"
# Output directory for results
OUTPUT_DIR = "../results" + "/Test"
OUTPUT_DIR_EDA = "../results" + "/Test_EDA"

SEED = 9626


# Default values used for the training process in the grid search and final training
GRID_TRAINING = {
    "max_epochs": 100,
    "patience": 8, 
    "warmup_epochs": 5
}
FINAL_TRAINING = {
    "max_epochs": 100, 
    "patience": 8, 
    "warmup_epochs": 5
}


# Common hyperparameters to the three architectures
BASE_HP = {
    "lr": 1e-3,
    "head_hidden": 128,
    "head_dropout": 0.3
}

# Config per architecture:
MODEL_SPECS = {
    "MLP": {
        # use_grid_search = False -> Uses base_hp and doesn't grid search
        "use_grid_search": True, # Tries all combinations in the grid
        # Fixed or final values chosen
        "base_hp": {
            **BASE_HP,
            "hidden_dim": 128,
            "rep_dim": 32,
            "dropout": 0.4,
            "weight_decay": 1e-5
        },
        # Grid of hyperparameters to search (when use_grid_search = True)
        "grid": {
            "hidden_dim": [64, 128],
            "rep_dim": [32, 64, 128],
            "dropout": [0.3, 0.4, 0.5],
            "weight_decay": [1e-5, 1e-4, 1e-3]
        }
    },
    "CNN": {
        "use_grid_search": True,
        "base_hp": {
            **BASE_HP,
            "channels": 128,
            "kernel_size": 5,
            "rep_dim": 64,
            "dropout": 0.3,
            "weight_decay": 1e-3
        },
        "grid": {
            "channels": [32, 64, 128],
            "kernel_size": [2, 3, 4, 5, 8, 12, 24],
            "dropout": [0.3, 0.4, 0.5],
            "weight_decay": [1e-4, 1e-3, 1.5e-3]
        }
    },
    "GRU": {
        "use_grid_search": True,
        "base_hp": {
            **BASE_HP,
            "hidden_dim": 64,
            "num_layers": 4,
            "rep_dim": 64,
            "dropout": 0.1,
            "weight_decay": 1.5e-5
        },
        "grid": {
            "hidden_dim": [64, 128],
            "num_layers": [2, 3, 4, 5, 6],
            "dropout": [0.1, 0.2, 0.3],
            "weight_decay": [1.5e-5, 1e-4, 1e-3]
        }
    }
}