# Vectors of the variables to predict based on their type
# Classification tasks
BINARY  = ["DMT2"] # Can be empty
# Regression tasks
NUMERICS = ["sf36", "sedentary", "bdi", "mmse", "chair_stand"] # can´t be empty

TARGET_COLS = BINARY + NUMERICS
TARGET_INDEX  = {task: i for i, task in enumerate(TARGET_COLS)}