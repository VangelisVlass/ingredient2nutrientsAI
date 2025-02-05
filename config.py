from log_manager import  EXPERIMENT_ID
import os

# 🚀 **CONFIGURATION**
DATA_DIR = r"C:\Users\vange\Desktop\FoodData_Central_branded_food_csv_2024-10-31"

CONFIG = {
    "experiment_id": EXPERIMENT_ID,
    "model_used": "distilbert-base-uncased",
    "tokenizer_used": "distilbert-base-uncased",
    "processed_data_file": "data.csv",
    "tokenized_data_file": "tokenized_data.csv",
    "long_ingredients_file": "long_ingredients.csv",
    "branded_food_file": os.path.join(DATA_DIR, "branded_food.csv"),
    "food_nutrient_file": os.path.join(DATA_DIR, "food_nutrient.csv"),
    "max_token_length": 256,
    "batch_size": 64,
    "epochs": 12,
    "learning_rates": [1.2e-4],
    "loss_function": "SmoothL1Loss",
    "nutrients_predicted": ["Protein", "Total Fat", "Sodium, Na", "Total Sugar", "Total Fiber"],
    "sample_size": 20000,
    "normalise": "none", # Options: "none", "Softmax"
    "hidden_layers": [ 512, 256, 128],
    "dropout_rate": 0.1,
    "activation_function": "gelu"  # Options: "gelu", "swish", "relu", "tanh", "sigmoid"
}