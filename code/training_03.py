import math
import torch
import torch.nn.functional as F
import pandas as pd

from config import BINARY, NUMERICS, TARGET_COLS, TASK_INDEX

# Calculate the mean of loss values, ignoring masked elements
def masked_mean(loss_values, mask):
    # Counts the amount of values where mask = 1 (0 doesn't add anyways)
    # IF mask.sum() = 0, it makes it 1 -> avoid division by zero
    denom = mask.sum().clamp_min(1.0)
    # Multiply each loss by its mask so that we eliminate cases where mask = 0
    # Sum the remaining values and divide by the amount of valid values
    return (loss_values * mask).sum() / denom # Returns the mean of the valid values


def multitask_loss(predicted_outcomes, real_outcomes, mask, task_weights = None):
    # If no weights have been provided, assign them a weight of 1.0
    if task_weights is None:
        task_weights = {task: 1.0 for task in TARGET_COLS} # All tasks are equally important 

    losses = {} # Initialize a dictionary to store the loss for each task

    # Binary classification task: Use Binary cross-entropy loss
    for task in BINARY:
        idx = TASK_INDEX[task]
        # Compute the binary cross-entropy loss between the predicted logits and the true labels for the current task
        # PD: logit = raw model output before applying the sigmoide function
        # Reduction = "none" -> The loss is computed for each sample without averaging
        bce = F.binary_cross_entropy_with_logits(predicted_outcomes[task], real_outcomes[:, idx], reduction = "none")
        # Calculate the mean of the valid losses (where mask = 1)
        losses[task] = masked_mean(bce, mask[:, idx]) # Store the mean loss for the current task in the dictionary

    # Numerical tasks: Use Mean Squared Error
    for task in NUMERICS:
        idx = TASK_INDEX[task]
        # Compute the mean squared error between the predicted values and the true values for the current task
        # ** 2 = Penalizes larger errors more heavily and prevents positive and negative errors from cancelling each other out
        mse = (predicted_outcomes[task] - real_outcomes[:, idx]) ** 2
        losses[task] = masked_mean(mse, mask[:, idx])

    # Calculate the total loss of the model: weighted sum of the individual task losses
    total = sum(task_weights[t] * losses[t] for t in TARGET_COLS)
    return total, losses


# 1 epoch has B batches, each batch has N samples
def train_one_epoch(model, loader, optimiser, device):
    model.train() # Model in training mode
    total_loss = 0.0 # Initialize the total loss for the epoch
    for batch in loader:
        x = batch["x"].to(device) # Matrix of hours x descriptors (sample, hours, descriptors)
        cov = batch["cov"].to(device) # Vector of clinical covariates (sample, covariates)
        y = batch["target"].to(device) # Real clinical targets (sample, targets)
        mask = batch["mask"].to(device) # Mask indicating which clinical targets exist

        # Restarts the gradients (saved by default)
        optimiser.zero_grad()
        # Forward pass: model predicts the outcomes based on the input data
        outputs = model(x, cov)
        # Calculates the loss for each task and the total loss (used for training)
        loss, losses = multitask_loss(outputs, y, mask)
        # Calculates the gradients (how much the weight should change to reduce the loss) through backpropagation
        loss.backward()
        # Prevents exploding gradients by scaling down the gradients if their norm > 1.0
        # clip_grad_norm_ is a function that modifies the gradients of the model's parameters in-place
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0) # Gradient Clipping
        # Updates the model's parameters based on the gradients:
        # Updates the weights (using Adam's optimizer)
        optimiser.step() # Learning step
        
        # loss.item() = scalar value of the loss tensor
        # x.size(0) = number of samples in the batch
        total_loss += loss.item() * x.size(0) # Accumulates the total loss for the epoch

    return total_loss / len(loader.dataset) # Mean loss of the total trainning set (for 1 epoch)


# # Disables gradient calculation
@torch.no_grad() # Gradients only needed during training
def evaluate_loss(model, loader, device):
    # Similar to train_one_epoch, but without backpropagation and weight updates
    model.eval()
    total_loss = 0.0
    for batch in loader:
        x = batch["x"].to(device)
        cov = batch["cov"].to(device)
        y = batch["target"].to(device)
        mask = batch["mask"].to(device)

        outputs = model(x, cov)
        loss, losses = multitask_loss(outputs, y, mask)
        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


# Main training function
def fit_model(model, train_data, val_data, device,
              lr = 1e-3, weight_decay = 1e-4, max_epochs = 100, patience = 10, warmup_epochs = 5):
    model = model.to(device)
    optimiser = torch.optim.Adam(
        model.parameters(), # trainable parameters of the model
        lr = lr, # learning rate
        weight_decay = weight_decay # L2 regularization -> prevents overfitting
    )

    # Initial linear scheduler
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimiser,
        start_factor = 0.1, # lr is just 10% of its original value during
        end_factor = 1.0,
        total_iters = warmup_epochs # the first "warmup_epochs" epochs
    )
    # Linear scheduler that reduces the learning rate if the metric stops improving
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode = "min", # the metric is expected to decrease (loss)
        factor = 0.5, # reduces the learning rate to half its value
        patience = 5, # wait 5 epoch before decreasing the metrics's value
        min_lr = 1e-5 # lr won't fall below 1e-5
    )

    # Initializacion of variables to track the best validation loss and model state
    best_val = float("inf") # Initializes in infinity so that any loss is better
    # Initializes in an initial copy of the weights of the model found during training
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    patience_counter = 0 # Initializes the counter for the amount of consecutive epochs the model has not improved
    history = [] # Initializes the list where the training history will be stored

    # Trains a "max_epochs" number of epochs (can stop early if patience is exceeded)
    for epoch in range(1, max_epochs + 1):
        # Training
        train_loss = train_one_epoch(model, train_data, optimiser, device)
        # Validation
        val_loss = evaluate_loss(model, val_data, device)
        # Get the current learning rate of the optimizer (from the first parameter group)
        current_lr = optimiser.param_groups[0]["lr"] # Saved to know the learning rate used in the epoch
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": current_lr})

        # if in the warmup phase:
        if epoch <= warmup_epochs:
            warmup_scheduler.step() # update the learning rate according to the warmup scheduler
        # if we're no longer in the warmup phase and val_loss is not NaN):
        elif not math.isnan(val_loss):
            plateau_scheduler.step(val_loss) # update the learning rate according to the plateau scheduler

        # If the validation loss has improved:
        if not math.isnan(val_loss) and val_loss < best_val:
            best_val = val_loss # update the best validation loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()} # save the model's state
            patience_counter = 0 # reset the patience counter because there was an improvement
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            break

    # Loads the best model state (weights) found during training
    model.load_state_dict(best_state)
    # Returns the best model and the training history (as a DataFrame)
    return model, pd.DataFrame(history)
