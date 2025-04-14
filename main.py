import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AdamW
import pandas as pd
import os
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from log_manager import (
    LOGS_DIR,
    calculate_percentage_differences,
    check_nutrient_ranges,
    log_correlation_matrix, 
    log_feature_importance, 
    log_loss_per_epoch, 
    log_message, 
    log_execution_summary, 
    log_misclassification_distribution, 
    log_training_results, 
    log_best_validation_results, 
    log_full_validation_predictions,
    save_experiment_results_to_csv,
    save_high_nutrient_rows)

from plot_manager import  (
    plot_actual_vs_predicted,
    plot_batch_size_vs_convergence, 
    plot_bland_altman, 
    plot_bland_altman_all,
    plot_correlation_matrix,
    plot_loss_function_comparison,
    plot_loss_per_epoch,
    plot_misclassification, 
    plot_nutrient_accuracy, 
    plot_percentage_difference_boxplot)
from process_data import process_data, save_branded_food_category
from tokenize_ingredients import tokenize_ingredients
from softmax import inverse_softmax
from model import ModifiedDistilBERT, get_loss_function
from config import CONFIG
from experiment import EXPERIMENT_ID

# 📂 **Ensure Logs Directory Exists**
os.makedirs("logs", exist_ok=True)

# 📝 **DATASET CLASS**
class FoodDataset(Dataset):
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        input_ids = torch.tensor(eval(row["input_ids"]), dtype=torch.long)
        attention_mask = torch.tensor(eval(row["attention_mask"]), dtype=torch.long)
        nutrients = torch.tensor(
            [row[nutrient] for nutrient in CONFIG["nutrients_predicted"]],
            dtype=torch.float,
        )
        return input_ids, attention_mask, nutrients

# 📊 **LOSS FUNCTION HANDLER**
def get_loss_function():
    loss_functions = {
        "MSELoss": nn.MSELoss(),
        "SmoothL1Loss": nn.SmoothL1Loss(),
    }
    return loss_functions.get(CONFIG["loss_function"], nn.MSELoss())

# 🔄 **DATA PREPARATION**
def prepare_data():
    log_message("🔄 Preparing data...")

    process_data(
        CONFIG["branded_food_file"],
        CONFIG["food_nutrient_file"],
        CONFIG["filtered_food_category"],
        CONFIG["processed_data_file"],
        CONFIG["sample_size"],
    )

    tokenize_ingredients(
        CONFIG["processed_data_file"],
        CONFIG["tokenized_data_file"],
        CONFIG["long_ingredients_file"],
        CONFIG["max_token_length"],
    )

# 🚀 **TRAINING FUNCTION**
def train_model():
    log_execution_summary(CONFIG)

    # 📂 **Load Dataset**
    dataset = FoodDataset(CONFIG["tokenized_data_file"])
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_message(f"🚀 Using device: {device}")

    best_val_loss = float("inf")
    best_lr = None
    best_epoch_predictions = None
    best_epoch_targets = None
    best_epoch_all_preds = None
    best_epoch_all_targets = None
    best_hidden_representations = None  # Stores the best epoch's hidden representations
    

    for lr in CONFIG["learning_rates"]:
        log_message(f"\n🚀 Training with learning rate: {lr}\n")

        model = ModifiedDistilBERT()
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=lr)
        loss_function = get_loss_function()
        best_model_state = None

        for epoch in range(CONFIG["epochs"]):
            model.train()
            total_train_loss = 0

            # 🔄 **Training Loop**
            for input_ids, attention_mask, nutrients in train_loader:
                input_ids, attention_mask, nutrients = (
                    input_ids.to(device),
                    attention_mask.to(device),
                    nutrients.to(device),
                )

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask, return_embedding=True)
                
                # Ensure correct unpacking based on model output
                if isinstance(outputs, tuple):
                    predictions, hidden_representations = outputs
                else:
                    predictions, hidden_representations = outputs, None

                loss = loss_function(predictions, nutrients)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            # 🔹 **Validation Phase**
            model.eval()
            total_val_loss = 0
            all_preds, all_targets, all_hidden_representations = [], [], []

            with torch.no_grad():
                for input_ids, attention_mask, nutrients in val_loader:
                    input_ids, attention_mask, nutrients = (
                        input_ids.to(device),
                        attention_mask.to(device),
                        nutrients.to(device),
                    )

                    outputs = model(input_ids, attention_mask, return_embedding=True)
                    
                    if isinstance(outputs, tuple):
                        predictions, hidden_representations = outputs
                    else:
                        predictions, hidden_representations = outputs, None

                    loss = loss_function(predictions, nutrients)

                    total_val_loss += loss.item()
                    all_preds.append(predictions.cpu().numpy())
                    all_targets.append(nutrients.cpu().numpy())

                    if hidden_representations is not None:
                        all_hidden_representations.append(hidden_representations.cpu().numpy())

            avg_val_loss = total_val_loss / len(val_loader)
            log_message(f"✅ Epoch {epoch + 1}/{CONFIG['epochs']} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            # 🔹 **Log Loss to CSV**
            log_loss_per_epoch(epoch + 1, avg_train_loss, avg_val_loss, CONFIG["experiment_id"])

            # Convert lists to NumPy arrays
            all_preds = np.concatenate(all_preds, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            if all_hidden_representations:
                all_hidden_representations = np.concatenate(all_hidden_representations, axis=0)
            else:
                all_hidden_representations = None

            # 🔄 **Apply inverse Softmax transformation if needed**
            if CONFIG["normalise"] == "Softmax":
                all_preds_original = inverse_softmax(all_preds)
                all_targets_original = inverse_softmax(all_targets)
            else:
                all_preds_original = all_preds
                all_targets_original = all_targets

            # 🔹 **Compute Performance Metrics**
            mae = mean_absolute_error(all_targets_original, all_preds_original)
            mse = mean_squared_error(all_targets_original, all_preds_original)
            r2 = r2_score(all_targets_original, all_preds_original)

            log_message(f"📊 Validation Metrics - MAE: {mae:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}")

            # 🔹 **Track Best Validation Results**
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_lr = lr
                best_model_state = model.state_dict()
                best_epoch_predictions = np.mean(all_preds_original, axis=0)
                best_epoch_targets = np.mean(all_targets_original, axis=0)
                best_epoch_all_preds = all_preds_original
                best_epoch_all_targets = all_targets_original
                best_hidden_representations = all_hidden_representations

        if best_model_state:
            model_path = f"logs/best_model_exp_{CONFIG['experiment_id']}_lr_{lr}.pth"
            torch.save(best_model_state, model_path)
            log_message(f"✅ Best model for LR {lr} saved to {model_path}")

            # ✅ Log Feature Importance using the best model
            best_model = ModifiedDistilBERT()
            best_model.load_state_dict(best_model_state)
            best_model.to(device)
            log_feature_importance(best_model, CONFIG["experiment_id"])

    # 🔹 **Log Best Validation Results (Averaged Values)**
    if best_epoch_predictions is not None and best_epoch_targets is not None:
        log_best_validation_results(best_epoch_targets, best_epoch_predictions, CONFIG["experiment_id"], CONFIG["nutrients_predicted"])

    # 🔹 **Log Full Validation Predictions (All Samples)**
    if best_epoch_all_preds is not None and best_epoch_all_targets is not None:
        log_full_validation_predictions(best_epoch_all_targets, best_epoch_all_preds, CONFIG["experiment_id"], CONFIG["nutrients_predicted"])

        # 🔥 Log Misclassification Rates
        log_misclassification_distribution(best_epoch_all_targets, best_epoch_all_preds, CONFIG["experiment_id"], CONFIG["nutrients_predicted"])

    # 🔹 **Log Correlation Matrix (AFTER TRAINING)**
    if best_hidden_representations is not None:
        log_correlation_matrix(best_hidden_representations, best_epoch_all_targets, CONFIG["experiment_id"], CONFIG["nutrients_predicted"])

    # 🔹 **Log Training Results**
    log_training_results(best_val_loss, avg_train_loss, best_lr, mae, mse, r2)


# 🚀 **MAIN EXECUTION**
def main():
    #prepare_data()
    #train_model()   
    plot_misclassification("3555bffe")
    
if __name__ == "__main__":
    main()
