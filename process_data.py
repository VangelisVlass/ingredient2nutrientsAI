import numpy as np
import pandas as pd
from log_manager import log_message
from softmax import softmax_normalization
from config import CONFIG

EXPERIMENTS_FILE = "experiments.json"

def process_data(branded_food_file, food_nutrient_file, output_file="data.csv", sample_size=5000):
    """
    Processes food data by merging branded food and nutrient information,
    computing the 'Others' column, applying softmax normalization, 
    and ensuring all values are per 100g/ml.
    """

    # Define relevant nutrients
    relevant_nutrients = {
        1003: "Protein", 1004: "Total Fat", 
        1093: "Sodium, Na", 2000: "Total Sugar", 1079: "Total Fiber"
    }

    log_message(f"ℹ️ Nutrients retained in dataset: {list(relevant_nutrients.values())}")
    log_message(f"🔹 Using sample size: {sample_size}")

    # Load and sample branded food data
    log_message("🔄 Loading and sampling branded food data...")
    branded_food = pd.read_csv(branded_food_file, usecols=["fdc_id", "ingredients"], low_memory=True)

    if len(branded_food) > sample_size:
        branded_food = branded_food.sample(n=sample_size, random_state=42)

    log_message(f"🔹 Sampled branded food entries: {len(branded_food)}")

    # Load and process nutrient data
    log_message("🔄 Loading nutrient data...")
    food_nutrient = pd.read_csv(food_nutrient_file, low_memory=True)

    # Filter for relevant nutrients
    log_message("🔄 Filtering relevant nutrients...")
    food_nutrient_filtered = food_nutrient[food_nutrient["nutrient_id"].isin(relevant_nutrients.keys())]

    # Aggregate and pivot nutrient data
    log_message("🔄 Aggregating nutrient data...")
    food_nutrient_filtered = food_nutrient_filtered.groupby(["fdc_id", "nutrient_id"], as_index=False).agg({"amount": "mean"})

    log_message("🔄 Pivoting nutrient data...")
    food_nutrient_pivoted = food_nutrient_filtered.pivot(index="fdc_id", columns="nutrient_id", values="amount").reset_index()
    food_nutrient_pivoted.columns = food_nutrient_pivoted.columns.astype(str)

    # Merge datasets
    log_message("🔄 Merging datasets...")
    merged_data = branded_food.merge(food_nutrient_pivoted, on="fdc_id", how="left").fillna(0)

    log_message(f"🔹 Merged dataset size: {len(merged_data)}")

    # Rename nutrient columns properly
    log_message("🔄 Renaming nutrient columns...")
    for nutrient_id, nutrient_name in relevant_nutrients.items():
        nutrient_id_str = str(nutrient_id)
        if nutrient_id_str in merged_data.columns:
            merged_data[nutrient_name] = merged_data[nutrient_id_str]

    # Convert Sodium from mg → g
    log_message("🔄 Converting Sodium from mg to g...")
    if "Sodium, Na" in merged_data.columns:
        merged_data["Sodium, Na"] /= 1000  

    # Keep Only Relevant Columns
    log_message("🔄 Filtering dataset to retain only relevant columns...")
    nutrient_cols = list(relevant_nutrients.values())
    columns_to_keep = ["fdc_id", "ingredients"] + nutrient_cols
    merged_data = merged_data[[col for col in columns_to_keep if col in merged_data.columns]]

    # Compute "Others" column correctly
    log_message("🔄 Calculating 'Others' column...")

    # Avoid computing 'Others' if all nutrients are already 100 (rare case)
    merged_data["Others"] = 100 - merged_data[nutrient_cols].sum(axis=1)
    merged_data["Others"] = merged_data["Others"].clip(lower=0)  # Ensure no negative values

    # Ensure all-zero nutrient rows get "Others" = 100
    all_zero_rows = merged_data[nutrient_cols].sum(axis=1) == 0
    merged_data.loc[all_zero_rows, "Others"] = 100

    if CONFIG["normalise"] == "Softmax":
        # Apply Softmax Normalization
        log_message("🔄 Applying Softmax Normalization to nutrient data...")

        # Define the columns that need normalization (excluding non-numeric columns)
        full_nutrient_cols = ["Protein", "Total Fat", "Sodium, Na", "Total Sugar", "Total Fiber", "Others"]

        # Select only numeric columns and ensure correct data types
        nutrient_data = merged_data[full_nutrient_cols].apply(pd.to_numeric, errors='coerce')

        # Handle NaN values before applying softmax
        nutrient_data = nutrient_data.fillna(0)  

        # Apply Softmax Normalization
        merged_data[full_nutrient_cols] = softmax_normalization(nutrient_data.values)

    # Save Processed Data
    log_message(f"✅ Saving processed dataset to '{output_file}'...")
    merged_data.to_csv(output_file, index=False, encoding="utf-8")

    log_message("✅ Data processing complete! 🚀")
