```text
BehaviouralPatterns_Movement/
├── code/
│   ├── config.py               # Configuration settings: targets, paths, hyper-parameters
│   ├── eda.py                  # Exploratory data analysis: statistics tables and visualisations
│   ├── preprocessing_01.py     # Data loading, splitting, normalisation, Dataset classes
│   ├── models_02.py            # Neural network encoders (MLP, CNN, GRU) + multi-task prediction head
│   ├── training_03.py          # Training loop, loss computation, learning rate schedulers, early stopping
│   ├── evaluation_04.py        # Performance metrics, prediction collection, participant-level bootstrap
│   ├── compare_models.py       # MAIN FUNCTION: performs grid search, training, and comparison
├── results/                    # Created automatically
│   ├── eda-1Week/              # Exploratory data analysis outputs
│   │   ├── table_descriptors_statistics.csv          # Summary statistics of the accelerometry descriptors
│   │   ├── table_covariates_targets_statistics.csv   # Summary statistics of the prediction targets
│   │   ├── correlation.png                           # Pearson correlation matrix heatmap
│   │   ├── demographics.png                          # Amount of measurement weeks
│   │   ├── diurnal_profile.png                       # Hourly averaged activity patterns across 24-hour cycle
│   │   └── outcome_distributions.png                 # Distribution histograms for all clinical outcomes
│   └── comparison-1Week/       # Model comparison outputs
│       ├── grid_search_*.csv                         # Hyperparameter grid search results per architecture
│       ├── best_hyperparameters.csv                  # Optimal hyperparameters for each model
│       ├── models_comparison.csv                     # Test set performance metrics with 95% confidence intervals
│       ├── loss_curves_comparison.png                # Training and validation loss trajectories
│       ├── metrics_comparison.png                    # Bar plots comparing primary metrics across models
│       └── pearson_correlation.png                   # Correlation coefficients for regression tasks
```

## Folder Structure

### `code/`
Contains the machine learning pipeline, organised by stage:

- **config.py**: Configuration file defining all hyperparameters, file paths, model specifications, and task definitions. Imported by all other modules to ensure consistency.

- **eda.py**: Generates exploratory data analysis tables and visualisations. Computes summary statistics for accelerometry descriptors and clinical variables. Creates correlation matrices, demographic distributions, diurnal activity profiles, and outcome histograms. Outputs tables as CSV files.

- **preprocessing_01.py**: Handles data loading from `.npy` and `.parquet` files, participant-level stratified splitting (70% training, 15% validation, 15% testing), z-score normalisation based on training set statistics, and creates PyTorch Dataset and DataLoader objects. Applies data augmentation during training.

- **models_02.py**: Defines three neural network encoder architectures (MLP, CNN, GRU) and a unified multi-task prediction head. The head handles both binary classification and continuous regression tasks simultaneously, with task-specific output layers.

- **training_03.py**: Implements the training loop with configurable loss functions (binary cross-entropy for classification, mean squared error for regression), learning rate scheduling, and early stopping based on validation loss. Returns training history and optimal model checkpoints.

- **evaluation_04.py**: Computes evaluation metrics (AUC-ROC for binary tasks, RMSE and Pearson correlation for regression tasks). Implements participant-level bootstrap resampling to estimate 95% confidence intervals.

- **compare_models.py**: Runs the entire pipeline. Performs grid search over hyperparameter combinations, trains models with optimal settings, evaluates performance, and generates comparative visualisations and summary tables.

### `results/`
Directory automatically created to store all analysis outputs:

#### `results/eda/`
Exploratory data analysis outputs:

- **table_descriptors_statistics.csv**: Hourly accelerometry descriptors with mean, standard deviation, minimum, and maximum values computed across all weeks.

- **table_covariates_targets_statistics.csv**: Clinical variables classified by type (covariate, regression target, binary classification target). Reports sample size, missing values, and summary statistics.

- **correlation.png**: Heatmap of Pearson correlation coefficients among clinical variables. Colour intensity indicates correlation strength; red indicates positive associations, blue indicates negative associations.

- **demographics.png**: Total number of weeks.

- **diurnal_profile.png**: Line plots showing the 24-hour averaged activity pattern. Displays mean ENMO (physical activity intensity) on the primary axis and sleep proportions on the secondary axis.

- **outcome_distributions.png**: Histograms for the prediction targets. Includes mean lines for continuous variables and count labels for the binary outcome.

#### `results/comparison/`
Model comparison and benchmarking outputs:

- **grid_search_*.csv**: One file per architecture (MLP, CNN, GRU) logging all hyperparameter combinations tested, validation loss, optimal epoch, and early stopping epoch.

- **best_hyperparameters.csv**: The single optimal hyperparameter configuration selected for each architecture based on minimum validation loss.

- **models_comparison.csv**: Test set performance across all tasks and models, including point estimates and 95% bootstrap confidence intervals for each metric.

- **loss_curves_comparison.png**: Three subplots showing training and validation loss evolution during final training for each architecture. Marks the epoch of best validation loss.

- **metrics_comparison.png**: Bar plots with error bars comparing the primary performance metric (AUC-ROC for classification, RMSE for regression) across the three models per task. The best-performing model per task is highlighted with a bold border.

- **pearson_correlation.png**: Bar plots with confidence intervals showing Pearson correlation between predicted and observed values for continuous regression tasks.