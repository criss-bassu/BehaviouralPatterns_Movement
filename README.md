```text
BehaviouralPatterns_Movement/
├── code/
│   ├── config.py               # all settings: targets, paths, hyper-parameters
│   ├── preprocessing_01.py     # data loading, splitting, normalisation, Dataset
│   ├── models_02.py            # the three encoders + multi-task head
│   ├── training_03.py          # training loop, loss, schedulers, early stopping
│   ├── evaluation_04.py        # metrics, predictions, participant bootstrap
│   ├── compare_models.py       # ENTRY POINT: orchestrates everything
├── data/
│   ├── tensor_X.npy            # the (N, 168, 14) input tensor
│   └── dfParticipants.parquet  # metadata + target columns
├── results/                    # created automatically; all outputs
```
