```text
BehaviouralPatterns_Movement/
├── code/
│   ├── config.py               # Configuration settings: targets, paths, hyper-parameters
│   ├── eda_01.py               # Exploratory data analysis: statistics tables and visualisations
│   ├── preprocessing_01.py     # Data loading, splitting, normalisation, Dataset classes
│   ├── models_02.py            # Neural network encoders (MLP, CNN, GRU) + multi-task prediction head
│   ├── training_03.py          # Training loop, loss computation, learning rate schedulers, early stopping
│   ├── evaluation_04.py        # Performance metrics, prediction collection, participant-level bootstrap
│   ├── compare_models.py       # ENTRY POINT: orchestrates grid search, training, and comparison
├── data/
│   ├── tensor_X.npy            # Accelerometry tensor of shape (N_weeks, 168_hours, 14_descriptors)
│   └── dfParticipants.parquet  # Participant metadata and clinical target variables
├── results/                    # Created automatically; contains all outputs organised by analysis type
│   ├── eda/                    # Exploratory data analysis outputs
│   │   ├── table_descriptors_statistics.csv          # Summary statistics of the 14 accelerometry descriptors
│   │   ├── table_covariates_targets_statistics.csv   # Summary statistics of covariates and prediction targets
│   │   ├── correlation.png                           # Pearson correlation matrix heatmap
│   │   ├── demographics.png                          # Distribution plots of age, BMI, and measurement weeks
│   │   ├── diurnal_profile.png                       # Hourly averaged activity patterns across 24-hour cycle
│   │   └── outcome_distributions.png                 # Distribution histograms for all clinical outcomes
│   └── comparison/             # Model comparison outputs
│       ├── grid_search_*.csv                         # Hyperparameter grid search results per architecture
│       ├── best_hyperparameters.csv                  # Optimal hyperparameters for each model
│       ├── models_comparison.csv                     # Test set performance metrics with 95% confidence intervals
│       ├── loss_curves_comparison.png                # Training and validation loss trajectories
│       ├── metrics_comparison.png                    # Bar plots comparing primary metrics across models
│       └── pearson_correlation.png                   # Correlation coefficients for regression tasks
```

## Folder Structure Explanation

### `code/`
Contains the complete machine learning pipeline, organised sequentially by stage:

- **config.py**: Central configuration file defining all hyperparameters, file paths, model specifications, and task definitions. Imported by all other modules to ensure consistency.

- **eda_01.py**: Generates exploratory data analysis tables and visualisations. Computes summary statistics for accelerometry descriptors and clinical variables. Creates correlation matrices, demographic distributions, diurnal activity profiles, and outcome histograms. Outputs tables as CSV files (not image files) for further analysis.

- **preprocessing_01.py**: Handles data loading from `.npy` and `.parquet` files, participant-level stratified splitting (70% training, 15% validation, 15% testing), z-score normalisation based on training set statistics, and creates PyTorch Dataset and DataLoader objects. Applies optional data augmentation during training.

- **models_02.py**: Defines three neural network encoder architectures (MLP, CNN, GRU) and a unified multi-task prediction head. The head handles both binary classification and continuous regression tasks simultaneously, with task-specific output layers.

- **training_03.py**: Implements the training loop with configurable loss functions (binary cross-entropy for classification, mean squared error for regression), learning rate scheduling, and early stopping based on validation loss. Returns training history and optimal model checkpoints.

- **evaluation_04.py**: Computes evaluation metrics (AUC-ROC for binary tasks, RMSE and Pearson correlation for regression tasks). Implements participant-level bootstrap resampling to estimate 95% confidence intervals, accounting for multiple measurements per individual.

- **compare_models.py**: Orchestrates the entire pipeline. Performs grid search over hyperparameter combinations, trains models with optimal settings, evaluates performance, and generates comparative visualisations and summary tables.

### `data/`
Contains preprocessed input data in efficient binary formats:

- **tensor_X.npy**: NumPy array of shape (5,420 weeks × 168 hours × 14 descriptors). Each row represents one week of accelerometry for one participant. The 14 descriptors include ENMO statistics (mean, SD, percentiles), sleep/bed/SIB proportions, step count, and ENMO intensity thresholds.

- **dfParticipants.parquet**: Pandas DataFrame with clinical and demographic information. Rows correspond to weeks in `tensor_X.npy`. Columns include participant identifiers, demographic covariates (age, BMI, sex), and outcome variables (DMT2 classification, continuous health measures: BDI, MMSE, SF-36, Chair Stand, Sedentary behaviour).

### `results/`
Directory automatically created to store all analysis outputs, organised by analysis type:

#### `results/eda/`
Exploratory data analysis outputs:

- **table_descriptors_statistics.csv**: Hourly accelerometry descriptors with mean, standard deviation, minimum, and maximum values computed across all 5,420 weeks.

- **table_covariates_targets_statistics.csv**: Clinical variables classified by type (covariate, regression target, binary classification target). Reports sample size, missing values, and summary statistics.

- **correlation.png**: Heatmap of Pearson correlation coefficients among clinical variables. Colour intensity indicates correlation strength; red indicates positive associations, blue indicates negative associations.

- **demographics.png**: Three histograms showing the age distribution, BMI distribution, and measurement duration (weeks per participant) across the study cohort.

- **diurnal_profile.png**: Line plots showing the 24-hour averaged activity pattern. Displays mean ENMO (physical activity intensity) on the primary axis and sleep/SIB (inactivity) proportions on the secondary axis, revealing distinct circadian rhythms.

- **outcome_distributions.png**: Histograms for the six prediction targets. Includes mean lines for continuous variables and count labels for the binary outcome (Type-2 diabetes prevalence).

#### `results/comparison/`
Model comparison and benchmarking outputs:

- **grid_search_*.csv**: One file per architecture (MLP, CNN, GRU) logging all hyperparameter combinations tested, validation loss, optimal epoch, and early stopping epoch.

- **best_hyperparameters.csv**: The single optimal hyperparameter configuration selected for each architecture based on minimum validation loss.

- **models_comparison.csv**: Test set performance across all tasks and models, including point estimates and 95% bootstrap confidence intervals for each metric.

- **loss_curves_comparison.png**: Three subplots showing training and validation loss evolution during final training for each architecture. Marks the epoch of best validation loss.

- **metrics_comparison.png**: Bar plots with error bars comparing the primary performance metric (AUC-ROC for classification, RMSE for regression) across the three models per task. The best-performing model per task is highlighted with a bold border.

- **pearson_correlation.png**: Bar plots with confidence intervals showing Pearson correlation between predicted and observed values for continuous regression tasks. Demonstrates model calibration on unseen test data.
