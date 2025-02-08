import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AdamW
import pandas as pd
import os
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from log_manager import log_message, log_execution_summary, log_training_results, log_best_validation_results, log_full_validation_predictions, EXPERIMENT_ID
from process_data import process_data, save_branded_food_category
from tokenize_ingredients import tokenize_ingredients
from softmax import inverse_softmax
from model import ModifiedDistilBERT 
from config import CONFIG


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

# 🎯 **TRAINING FUNCTION**
def train_model():
    log_execution_summary(CONFIG)

    dataset = FoodDataset(CONFIG["tokenized_data_file"])
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_message(f"🚀 Using device: {device}")

    best_overall_loss = float("inf")
    best_lr = None
    best_epoch_predictions = None
    best_epoch_targets = None
    best_epoch_all_preds = None
    best_epoch_all_targets = None

    for lr in CONFIG["learning_rates"]:
        log_message(f"\n🚀 Training with learning rate: {lr}\n")

        model = ModifiedDistilBERT()
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=lr)
        loss_function = get_loss_function()

        best_val_loss = float("inf")
        best_model_state = None

        for epoch in range(CONFIG["epochs"]):
            model.train()
            total_train_loss = 0

            for input_ids, attention_mask, nutrients in train_loader:
                input_ids, attention_mask, nutrients = (
                    input_ids.to(device),
                    attention_mask.to(device),
                    nutrients.to(device),
                )

                optimizer.zero_grad()
                predictions = model(input_ids, attention_mask)
                loss = loss_function(predictions, nutrients)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            # 🔹 **Validation Phase**
            model.eval()
            total_val_loss = 0
            all_preds, all_targets = [], []

            with torch.no_grad():
                for input_ids, attention_mask, nutrients in val_loader:
                    input_ids, attention_mask, nutrients = (
                        input_ids.to(device),
                        attention_mask.to(device),
                        nutrients.to(device),
                    )

                    predictions = model(input_ids, attention_mask)
                    loss = loss_function(predictions, nutrients)
                    total_val_loss += loss.item()

                    all_preds.append(predictions.cpu().numpy())
                    all_targets.append(nutrients.cpu().numpy())

            avg_val_loss = total_val_loss / len(val_loader)
            log_message(f"✅ Epoch {epoch + 1}/{CONFIG['epochs']} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            all_preds = np.concatenate(all_preds, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)

            # 🔄 **Apply inverse Softmax transformation**
            if CONFIG["normalise"] == "Softmax":
                all_preds_original = inverse_softmax(all_preds)
                all_targets_original = inverse_softmax(all_targets)
            else:
                all_preds_original = all_preds
                all_targets_original = all_targets

            # 🔹 **Compute Metrics**
            mae = mean_absolute_error(all_targets_original, all_preds_original)
            mse = mean_squared_error(all_targets_original, all_preds_original)
            r2 = r2_score(all_targets_original, all_preds_original)

            log_message(f"📊 Validation Metrics - MAE: {mae:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}")

            # 🔹 **Track Best Validation Results**
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict()
                best_epoch_predictions = np.mean(all_preds_original, axis=0)  # Average predictions
                best_epoch_targets = np.mean(all_targets_original, axis=0)  # Average actual values
                best_epoch_all_preds = all_preds_original  # Store all sample predictions
                best_epoch_all_targets = all_targets_original  # Store all sample targets

        if best_model_state:
            model_path = f"logs/best_model_exp_{EXPERIMENT_ID}_lr_{lr}.pth"
            torch.save(best_model_state, model_path)
            log_message(f"✅ Best model for LR {lr} saved to {model_path}")

        if best_val_loss < best_overall_loss:
            best_overall_loss = best_val_loss
            best_lr = lr

    # 🔹 **Log Best Validation Results (Averaged Values)**
    if best_epoch_predictions is not None and best_epoch_targets is not None:
        log_best_validation_results(best_epoch_targets, best_epoch_predictions, EXPERIMENT_ID, CONFIG["nutrients_predicted"])

    # 🔹 **Log Full Validation Predictions (All Samples)**
    if best_epoch_all_preds is not None and best_epoch_all_targets is not None:
        log_full_validation_predictions(best_epoch_all_targets, best_epoch_all_preds, EXPERIMENT_ID, CONFIG["nutrients_predicted"])

    # 🔹 **Log Training Results**
    log_training_results(best_overall_loss, avg_train_loss, best_lr, mae, mse, r2)



# 🚀 **MAIN EXECUTION**
def main():
    prepare_data()
    train_model()


if __name__ == "__main__":
    main()
