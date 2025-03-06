import datetime
import uuid
import json
import csv
import os

import numpy as np
import pandas as pd
import torch

from config import CONFIG
from experiment import EXPERIMENT_ID

# Define log files
LOGS_DIR = "logs"
PROCESS_LOG_FILE = os.path.join(LOGS_DIR, "process_log.txt")
TRAINING_LOG_FILE = os.path.join(LOGS_DIR, "training_log.txt")
EXPERIMENT_METADATA_FILE = os.path.join(LOGS_DIR, "experiments.json")
TRAINING_RESULTS_FILE = os.path.join(LOGS_DIR, "training_results.csv")

# Ensure log directory exists
os.makedirs(LOGS_DIR, exist_ok=True)


def log_message(message, log_type="process"):
    """
    Logs messages to either process_log.txt or training_log.txt.

    Args:
        message (str): The message to log.
        log_type (str): "process" for process logs, "training" for training logs.
    """
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    log_entry = f"{timestamp} [EXP-{EXPERIMENT_ID}] {message}"

    # Choose the correct log file
    log_file = PROCESS_LOG_FILE if log_type == "process" else TRAINING_LOG_FILE

    print(log_entry)  # Print to console

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")


def log_execution_summary(config):
    """
    Logs execution settings at the beginning of training and saves them in a structured JSON format.

    Args:
        config (dict): Dictionary containing all the settings (model, tokenizer, learning rates, etc.).
    """
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        **config  # Spread dictionary values
    }

    log_message(f"✅ Execution summary saved for EXP-{EXPERIMENT_ID}", "training")

    # Save summary to training log file
    with open(TRAINING_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, indent=4) + "\n\n")

    # Save structured experiment metadata
    try:
        with open(EXPERIMENT_METADATA_FILE, "r", encoding="utf-8") as f:
            experiments = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        experiments = {}

    experiments[EXPERIMENT_ID] = summary

    with open(EXPERIMENT_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(experiments, f, indent=4)

def log_training_results(best_val_loss, final_train_loss, learning_rate, mae, mse, r2_score):
    """
    Logs training results, updates experiment metadata, and stores them in a CSV file.

    Args:
        best_val_loss (float): The lowest validation loss achieved.
        final_train_loss (float): The final training loss at the last epoch.
        learning_rate (float): The learning rate used.
        mae (float): Mean Absolute Error (validation set).
        mse (float): Mean Squared Error (validation set).
        r2 (float): R² score (validation set).
    """
    results = {
        "experiment_id": EXPERIMENT_ID,
        "learning_rate": learning_rate,
        "best_validation_loss": best_val_loss,
        "final_train_loss": final_train_loss,
        "mae": mae,
        "mse": mse,
        "r2_score": r2_score
    }

    # Append results to training log file
    with open(TRAINING_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(results, indent=4) + "\n\n")

    # Update experiment metadata JSON
    try:
        with open(EXPERIMENT_METADATA_FILE, "r", encoding="utf-8") as f:
            experiments = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        experiments = {}

    if EXPERIMENT_ID in experiments:
        experiments[EXPERIMENT_ID]["training_results"] = results
    else:
        experiments[EXPERIMENT_ID] = {"training_results": results}

    with open(EXPERIMENT_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(experiments, f, indent=4)

    log_message(f"🏆 Training results saved for EXP-{EXPERIMENT_ID}", "training")

    # Save to CSV for easier visualization
    write_csv_training_results(results)

def write_csv_training_results(results):
    """
    Appends training results to a CSV file.

    Args:
        results (dict): Dictionary containing training metrics.
    """
    file_exists = os.path.isfile(TRAINING_RESULTS_FILE)
    
    with open(TRAINING_RESULTS_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)

        # Write header if file doesn't exist
        if not file_exists:
            writer.writerow(["experiment_id", "learning_rate", "best_validation_loss", 
                             "final_train_loss", "mae", "mse", "r2_score"])

        writer.writerow([
            results["experiment_id"], results["learning_rate"], results["best_validation_loss"],
            results["final_train_loss"], results["mae"], results["mse"], results["r2_score"]
        ])
    
    log_message(f"📊 Training results appended to {TRAINING_RESULTS_FILE}")

def log_best_validation_results(targets, predictions, experiment_id, nutrient_names):
    """
    Logs the best validation results to a CSV file with proper column names.
    
    Args:
        targets (np.array): Ground truth values.
        predictions (np.array): Model predictions.
        experiment_id (str): Unique experiment identifier.
        nutrient_names (list): List of nutrient names (e.g., ["Protein", "Total Fat", ...]).
    """
    os.makedirs(LOGS_DIR, exist_ok=True)  # Ensure the directory exists
    log_file = os.path.join(LOGS_DIR, f"best_validation_results_{experiment_id}.csv")

    # Define CSV Header
    headers = ["Nutrient"] + ["Target"] + ["Prediction"]

    # Open file in write mode (overwrite old file each time)
    with open(log_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)  # Write header

        # Write best validation results
        for i, nutrient in enumerate(nutrient_names):
            writer.writerow([nutrient, targets[i], predictions[i]])

def log_full_validation_predictions(targets, predictions, experiment_id, nutrient_names):
    """
    Logs all validation predictions to a CSV file, storing full validation results.
    
    Args:
        targets (np.array): Ground truth values.
        predictions (np.array): Model predictions.
        experiment_id (str): Unique experiment identifier.
        nutrient_names (list): List of nutrient names (e.g., ["Protein", "Total Fat", ...]).
    """
    os.makedirs(LOGS_DIR, exist_ok=True)  # Ensure the directory exists
    log_file = os.path.join(LOGS_DIR, f"validation_predictions_{experiment_id}.csv")

    # Define CSV Header
    headers = nutrient_names + [f"Predicted_{name}" for name in nutrient_names]

    # Open file in write mode
    with open(log_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)  # Write header

        # Write validation results for all samples
        for true_vals, pred_vals in zip(targets, predictions):
            writer.writerow(list(true_vals) + list(pred_vals))

    log_message(f"✅ Full validation predictions saved to {log_file}")


def log_loss_per_epoch(epoch, train_loss, val_loss, experiment_id):
    """
    Logs training and validation loss per epoch to a CSV file.

    Args:
        epoch (int): Current epoch number.
        train_loss (float): Training loss for this epoch.
        val_loss (float): Validation loss for this epoch.
        experiment_id (str): Unique experiment identifier.
    """
    os.makedirs(LOGS_DIR, exist_ok=True)  # Ensure logs directory exists
    log_file = os.path.join(LOGS_DIR, f"loss_per_epoch_{experiment_id}.csv")

    # Check if file exists to write headers
    file_exists = os.path.isfile(log_file)

    with open(log_file, mode="a", newline="") as f:
        writer = csv.writer(f)

        # Write header only if the file is new
        if not file_exists:
            writer.writerow(["Epoch", "Train Loss", "Validation Loss"])

        writer.writerow([epoch, train_loss, val_loss])


def log_feature_importance(model, experiment_id):
    """
    Logs feature importance based on the weight coefficients of the LAST Linear layer in MLP.
    
    Args:
        model (torch.nn.Module): The trained model.
        experiment_id (str): Unique experiment identifier.
    """
    os.makedirs(LOGS_DIR, exist_ok=True)  # Ensure logs directory exists
    log_file = os.path.join(LOGS_DIR, f"feature_importance_{experiment_id}.csv")

    # 🔍 **Find Last Linear Layer in MLP**
    last_linear_layer = None
    for layer in reversed(model.feedforward):  # Iterate in reverse order
        if isinstance(layer, torch.nn.Linear):
            last_linear_layer = layer
            break  # Capture only the last Linear layer

    if last_linear_layer is None:
        print("⚠️ No linear layer found in feedforward network.")
        return

    # 🔹 **Extract Weights from Last Linear Layer**
    weights = last_linear_layer.weight.detach().cpu().numpy()  # Shape: (num_nutrients, num_features)
    
    # 🔹 **Ensure Feature & Nutrient Mapping**
    num_nutrients, num_features = weights.shape
    nutrients = CONFIG["nutrients_predicted"]  # Should be 5 nutrients
    
    if num_nutrients != len(nutrients):
        print(f"⚠️ Mismatch: Model outputs {num_nutrients} nutrients, but CONFIG has {len(nutrients)}")
        return

    # 🔹 **Normalize Feature Importance**
    feature_importance = np.abs(weights)  # Absolute values of weight coefficients
    feature_importance = feature_importance / feature_importance.sum(axis=1, keepdims=True)  # Normalize row-wise

    # 🔹 **Save to CSV in Heatmap Format**
    with open(log_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        
        # Write Header (Nutrient Names)
        writer.writerow(["Feature_Index"] + nutrients)
        
        # Write Feature Importance Values
        for feature_idx in range(num_features):
            row = [feature_idx] + list(feature_importance[:, feature_idx])  # Feature importance per nutrient
            writer.writerow(row)

    print(f"✅ Feature importance logged in {log_file}")



def log_correlation_matrix(hidden_representations, targets, experiment_id, nutrient_names):
    """
    Logs the correlation matrix between the last hidden MLP representations and nutrient values.

    Args:
        hidden_representations (np.array): Feature representations before the final prediction layer.
        targets (np.array): Ground truth nutrient values (shape: [samples, num_nutrients]).
        experiment_id (str): Unique experiment identifier.
        nutrient_names (list): List of nutrient names.
    """
    os.makedirs(LOGS_DIR, exist_ok=True)  # Ensure directory exists
    log_file = os.path.join(LOGS_DIR, f"correlation_matrix_{experiment_id}.csv")

    # 🔹 **Convert to DataFrames**
    hidden_df = pd.DataFrame(hidden_representations, columns=[f"Feature_{i}" for i in range(hidden_representations.shape[1])])
    targets_df = pd.DataFrame(targets, columns=nutrient_names)

    # 🔹 **Combine Features & Nutrients**
    combined_df = pd.concat([hidden_df, targets_df], axis=1)

    # 🔹 **Compute Correlation Matrix**
    correlation_matrix = combined_df.corr()

    # 🔹 **Save to CSV**
    correlation_matrix.to_csv(log_file)

    print(f"✅ Correlation matrix saved to {log_file}")




def log_misclassification_distribution(targets, predictions, experiment_id, nutrient_names, threshold=0.1):
    """
    Logs the misclassification distribution based on a threshold.

    Args:
        targets (np.array): Ground truth nutrient values (shape: [samples, num_nutrients]).
        predictions (np.array): Model predictions (same shape as targets).
        experiment_id (str): Unique experiment identifier.
        nutrient_names (list): List of nutrient names.
        threshold (float): Error threshold (default: 10% of actual value).
    """
    os.makedirs(LOGS_DIR, exist_ok=True)  # Ensure the directory exists
    log_file = os.path.join(LOGS_DIR, f"misclassification_{experiment_id}.csv")

    # 🔹 **Compute Misclassification Rates**
    abs_error = np.abs(predictions - targets)
    relative_error = abs_error / (targets + 1e-8)  # Avoid division by zero

    # Count misclassified samples per nutrient
    misclassified = (relative_error > threshold).sum(axis=0)
    total_samples = targets.shape[0]
    misclassification_rates = misclassified / total_samples  # Compute percentage

    # Write to CSV
    with open(log_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Nutrient", "Misclassification_Rate"])

        for i, nutrient in enumerate(nutrient_names):
            writer.writerow([nutrient, misclassification_rates[i]])

    print(f"✅ Misclassification distribution logged in {log_file}")

