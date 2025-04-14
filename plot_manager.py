import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,r2_score

# Ensure directories exist
PLOTS_DIR = "plots"
LOGS_DIR = "logs"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ✅ Utility function for checking file existence
def check_file_exists(file_path, description):
    if not os.path.exists(file_path):
        print(f"⚠️ {description} file not found: {file_path}")
        return False
    return True

# ✅ Utility function for normalizing DataFrame column names
def normalize_columns(df):
    return df.rename(columns=lambda x: x.strip().lower().replace(" ", "_"))


### 🔥 **1. Heatmap – Feature Importance in Nutrient Prediction**
def plot_feature_importance(exp_id):
    file_path = os.path.join(LOGS_DIR, f"feature_importance_{exp_id}.csv")
    if not check_file_exists(file_path, "Feature Importance"):
        return

    df = pd.read_csv(file_path, index_col=0)  # Feature_Index is the index

    plt.figure(figsize=(12, 6))
    sns.heatmap(df, annot=False, cmap="coolwarm", fmt=".2f", cbar=True)

    plt.title("Feature Importance in Nutrient Prediction")
    plt.xlabel("Nutrients")
    plt.ylabel("Extracted Features")

    plot_filename = os.path.join(PLOTS_DIR, f"feature_importance_heatmap_{exp_id}.png")
    plt.savefig(plot_filename)
    plt.close()

    print(f"✅ Saved feature importance heatmap: {plot_filename}")


### 📉 **2. Training Convergence Plot – Loss Function per Epoch**
def plot_loss_per_epoch(exp_id):
    file_path = os.path.join(LOGS_DIR, f"loss_per_epoch_{exp_id}.csv")
    if not check_file_exists(file_path, "Loss per Epoch"):
        return

    df = pd.read_csv(file_path)
    df = normalize_columns(df)

    required_columns = {"epoch", "train_loss", "validation_loss"}
    if not required_columns.issubset(df.columns):
        print(f"⚠️ Missing required columns in CSV! Found: {df.columns.tolist()}")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], label="Training Loss", color="red")
    plt.plot(df["epoch"], df["validation_loss"], label="Validation Loss", color="blue")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Training Convergence - {exp_id}")
    plt.legend()

    plot_filename = os.path.join(PLOTS_DIR, f"plot_loss_per_epoch_{exp_id}.png")
    plt.savefig(plot_filename)
    plt.close()

    print(f"✅ Saved loss plot: {plot_filename}")


### 🔗 **3. Correlation Matrix – Relationship Between Features & Nutrients**
def plot_correlation_matrix(exp_id):
    file_path = os.path.join(LOGS_DIR, f"correlation_matrix_{exp_id}.csv")
    if not check_file_exists(file_path, "Correlation Matrix"):
        return

    df = pd.read_csv(file_path, index_col=0)

    plt.figure(figsize=(10, 8))

    # Explicitly set cmap to ensure a colorful heatmap
    sns.heatmap(df, annot=True, cmap="viridis", center=0, fmt=".2f", cbar=True)

    plt.title("Correlation Matrix – Extracted Features vs. Nutrients")

    plot_filename = os.path.join(PLOTS_DIR, f"correlation_matrix_{exp_id}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved correlation matrix heatmap: {plot_filename}")


### 📊 **4. Bar Chart – Misclassification Distribution Across Nutrient Categories**
def plot_misclassification(exp_id):
    file_path = os.path.join(LOGS_DIR, f"misclassification_{exp_id}.csv")
    if not check_file_exists(file_path, "Misclassification"):
        return

    df = pd.read_csv(file_path)
    df = normalize_columns(df)

    required_columns = {"nutrient", "misclassification_rate"}
    if not required_columns.issubset(df.columns):
        print(f"⚠️ Missing required columns! Found: {df.columns.tolist()}")
        return

    plt.figure(figsize=(8, 5))
    sns.barplot(x=df["nutrient"], y=df["misclassification_rate"], palette="viridis")

    plt.xlabel("Nutrient")
    plt.ylabel("Misclassification Rate (%)")
    plt.title(f"Misclassification Distribution - {exp_id}")
    plt.xticks(rotation=45)

    plot_filename = os.path.join(PLOTS_DIR, f"plot_misclassification_{exp_id}.png")
    plt.savefig(plot_filename)
    plt.close()

    print(f"✅ Saved misclassification plot: {plot_filename}")


### 📊 **5. Bar Chart – Actual vs Predicted Nutrient Values**
def plot_predicted_vs_actual(exp_id):
    file_path = os.path.join(LOGS_DIR, f"validation_predictions_{exp_id}.csv")
    if not check_file_exists(file_path, "Validation Predictions"):
        return

    df = pd.read_csv(file_path)

    nutrients = ["Protein", "Total Fat", "Sodium, Na", "Total Sugar", "Total Fiber"]
    predicted_columns = [f"Predicted_{n}" for n in nutrients]

    for nutrient, pred_nutrient in zip(nutrients, predicted_columns):
        plt.figure(figsize=(12, 6))

        max_value = max(df[nutrient].max(), df[pred_nutrient].max()) + 1e-5  # Scale adjustment
        actual_scaled = df[nutrient] / max_value
        predicted_scaled = df[pred_nutrient] / max_value

        indices = range(len(df))
        plt.bar(indices, actual_scaled, width=0.4, label="Actual", color="blue", alpha=0.6)
        plt.bar(indices, predicted_scaled, width=0.4, label="Predicted", color="orange", alpha=0.6)

        plt.xlabel("Sample Index")
        plt.ylabel(f"Normalized Nutrient Value (Scaled to Max={max_value:.2f})")
        plt.title(f"{nutrient}: Actual vs Predicted")
        plt.legend()

        plot_filename = os.path.join(PLOTS_DIR, f"plot_{nutrient.lower()}_actual_vs_predicted_{exp_id}.png")
        plt.savefig(plot_filename)
        plt.close()

        print(f"✅ Saved plot: {plot_filename}")


### 📊 **6. Bar Chart – Nutrient Prediction Accuracy**
def plot_nutrient_accuracy(exp_id):
    file_path = os.path.join(LOGS_DIR, f"best_validation_results_{exp_id}.csv")
    if not check_file_exists(file_path, "Best Validation Results"):
        return

    df = pd.read_csv(file_path)

    nutrients = df["Nutrient"].tolist()
    actual_values = df["Target"].tolist()
    predicted_values = df["Prediction"].tolist()

    mae_values = [mean_absolute_error([actual], [predicted]) for actual, predicted in zip(actual_values, predicted_values)]

    plt.figure(figsize=(10, 5))
    plt.bar(nutrients, mae_values, color="orange", alpha=0.7)

    plt.xlabel("Nutrients")
    plt.ylabel("Mean Absolute Error (MAE)")
    plt.title("Nutrient Prediction Accuracy (Lower MAE is Better)")
    plt.xticks(rotation=30)

    plot_filename = os.path.join(PLOTS_DIR, f"plot_accuracy_{exp_id}.png")
    plt.savefig(plot_filename)
    plt.close()

    print(f"✅ Saved accuracy plot: {plot_filename}")


def plot_bland_altman(exp_id, nutrient):
    """
    Generates a Bland-Altman plot for actual vs predicted nutrient values.

    Args:
        exp_id (str): The experiment ID used for log files.
        nutrient (str): The nutrient to analyze (e.g., "Protein").
    
    Returns:
        None
    """
    file_path = os.path.join(LOGS_DIR, f"validation_predictions_{exp_id}.csv")

    if not os.path.exists(file_path):
        print(f"⚠️ File not found: {file_path}")
        return

    df = pd.read_csv(file_path)

    # Ensure the nutrient and predicted columns exist
    predicted_nutrient = f"Predicted_{nutrient}"
    if nutrient not in df.columns or predicted_nutrient not in df.columns:
        print(f"⚠️ Missing columns for {nutrient} analysis.")
        return

    # Calculate Mean and Differences
    mean_values = (df[nutrient] + df[predicted_nutrient]) / 2
    differences =  df[predicted_nutrient]-  df[nutrient] 

    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    upper_limit = mean_diff + 1.96 * std_diff
    lower_limit = mean_diff - 1.96 * std_diff

    # Plot Bland-Altman
    plt.figure(figsize=(8, 6))
    plt.scatter(mean_values, differences, alpha=0.5, color="green", label="Data Points")
    plt.axhline(mean_diff, color='blue', linestyle='-', label=f"Mean ({mean_diff:.2f})")
    plt.axhline(upper_limit, color='brown', linestyle='--', label=f"+1.96 SD ({upper_limit:.2f})")
    plt.axhline(lower_limit, color='brown', linestyle='--', label=f"-1.96 SD ({lower_limit:.2f})")

    # Add value labels on the right side
    plt.text(np.max(mean_values), upper_limit, f"{upper_limit:.2f}", 
             verticalalignment='bottom', horizontalalignment='right', fontsize=10, color="brown")
    
    plt.text(np.max(mean_values), mean_diff, f"Mean\n{mean_diff:.2f}", 
             verticalalignment='bottom', horizontalalignment='right', fontsize=10, color="blue")

    plt.text(np.max(mean_values), lower_limit, f"{lower_limit:.2f}", 
             verticalalignment='top', horizontalalignment='right', fontsize=10, color="brown")

    # Labels and Title
    plt.xlabel(f"Mean of Actual & Predicted {nutrient}")
    plt.ylabel(f"Difference (Predicted - Actual) {nutrient}")
    plt.title(f"Bland-Altman Plot for {nutrient}")
    plt.legend()
    plt.grid(True)

    # Save plot
    plot_filename = os.path.join(PLOTS_DIR, f"bland_altman_{nutrient}_{exp_id}.png")
    plt.savefig(plot_filename)
    plt.show()

    print(f"✅ Bland-Altman plot saved: {plot_filename}")


def plot_bland_altman_all(exp_id, config):
    """
    Generates Bland-Altman plots for all nutrients in the config file with numerical annotations.

    Args:
        exp_id (str): Experiment ID.
        config (dict): Configuration dictionary containing nutrient names.
    """
    file_path = os.path.join(LOGS_DIR, f"validation_predictions_{exp_id}.csv")

    if not os.path.exists(file_path):
        print(f"⚠️ File not found: {file_path}")
        return

    # Load data
    df = pd.read_csv(file_path)

    for nutrient in config["nutrients_predicted"]:
        predicted_col = f"Predicted_{nutrient}"
        
        if nutrient not in df.columns or predicted_col not in df.columns:
            print(f"⚠️ {nutrient} or {predicted_col} not found in CSV. Skipping...")
            continue
        
        # Calculate Bland-Altman components
        actual_values = df[nutrient]
        predicted_values = df[predicted_col]
        mean_values = (actual_values + predicted_values) / 2
        differences = predicted_values - actual_values
        mean_diff = np.mean(differences)
        std_diff = np.std(differences)
        upper_limit = mean_diff + 1.96 * std_diff
        lower_limit = mean_diff - 1.96 * std_diff

        # Plot Bland-Altman
        plt.figure(figsize=(8, 5))
        plt.scatter(mean_values, differences, alpha=0.5, label="Differences", color="lightblue", edgecolors="black")

        # Mean and Limits
        plt.axhline(mean_diff, color='navy', linestyle='-', label="Mean")
        plt.axhline(upper_limit, color='brown', linestyle='--', label="+1.96 SD")
        plt.axhline(lower_limit, color='brown', linestyle='--', label="-1.96 SD")

        # Add numerical annotations
        plt.text(np.max(mean_values), upper_limit, f"{upper_limit:.1f}", 
            verticalalignment='bottom', horizontalalignment='right', fontsize=10, color="brown")

        plt.text(np.max(mean_values), mean_diff, f"{mean_diff:.1f}", 
            verticalalignment='bottom', horizontalalignment='right', fontsize=10, color="navy")

        plt.text(np.max(mean_values), lower_limit, f"{lower_limit:.1f}", 
            verticalalignment='top', horizontalalignment='right', fontsize=10, color="brown")

        plt.xlabel("Mean of Actual and Predicted")
        plt.ylabel("Difference (Predicted - Actual)")
        plt.title(f"Bland-Altman Plot for {nutrient}")
        plt.legend()
        plt.grid(True)

        # Save plot
        plot_filename = os.path.join(PLOTS_DIR, f"bland_altman_{nutrient}_{exp_id}.png")
        plt.savefig(plot_filename)
        plt.close()

        print(f"✅ Saved Bland-Altman plot: {plot_filename}")

def plot_percentage_difference_boxplot(exp_id):
    """
    Generates a box plot using Matplotlib for percentage differences of all nutrients, displaying outliers.

    Args:
        exp_id (str): The experiment ID used for log files.
    
    Returns:
        None
    """
    file_path = os.path.join(LOGS_DIR, f"percentage_differences_{exp_id}.csv")

    if not os.path.exists(file_path):
        print(f"⚠️ File not found: {file_path}")
        return

    # Load CSV
    df = pd.read_csv(file_path)

    # Select only the percentage difference columns (columns that contain "Diff_")
    diff_columns = [col for col in df.columns if "Diff_" in col]

    if not diff_columns:
        print(f"⚠️ No percentage difference columns found in {file_path}.")
        return

    # Extract data for plotting
    data = [df[col].dropna() for col in diff_columns]  # Remove NaN values per nutrient
    labels = [col.replace("Diff_", "").replace(" (%)", "") for col in diff_columns]  # Clean labels

    # Plot setup
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=labels, patch_artist=True, showfliers=True)  # Enable outliers

    # Labels and title
    plt.xticks(rotation=45)
    plt.xlabel("Nutrient")
    plt.ylabel("Percentage Difference (%)")
    plt.title(f"Box Plot of Percentage Differences per Nutrient (Exp: {exp_id})")

    # Save plot
    plot_filename = os.path.join(PLOTS_DIR, f"percentage_difference_boxplot_{exp_id}.png")
    plt.savefig(plot_filename)

    print(f"✅ Percentage difference box plot saved: {plot_filename}")

def plot_actual_vs_predicted(exp_id, CONFIG):
    """
    Generates separate scatter plots of actual vs. predicted values for each nutrient.

    Args:
        exp_id (str): The experiment ID used for log files.
        CONFIG (dict): Configuration containing nutrient names.

    Returns:
        None
    """
    file_path = os.path.join(LOGS_DIR, f"validation_predictions_{exp_id}.csv")

    if not os.path.exists(file_path):
        print(f"⚠️ File not found: {file_path}")
        return

    # Load Data
    df = pd.read_csv(file_path)

    # Extract nutrients from CONFIG
    nutrients = CONFIG["nutrients_predicted"]

    for nutrient in nutrients:
        predicted_nutrient = f"Predicted_{nutrient}"

        if nutrient not in df.columns or predicted_nutrient not in df.columns:
            print(f"⚠️ Missing columns for {nutrient}. Skipping.")
            continue

        actual = df[nutrient].values
        predicted = df[predicted_nutrient].values

        # Compute R² score
        r2 = r2_score(actual, predicted)

        # Create a new figure for each nutrient
        plt.figure(figsize=(6, 6))

        # Scatter Plot
        plt.scatter(actual, predicted, alpha=0.5, color="blue", label=f"Data Points ({nutrient})")

        # Identity Line (Ideal Prediction)
        min_val, max_val = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
        plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2, label="Ideal Line (y=x)")

        # Labels and Title
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"{nutrient} (R²: {r2:.3f})")
        plt.legend()
        plt.grid(True)

        # Save each plot separately
        plot_filename = os.path.join(PLOTS_DIR, f"scatter_actual_vs_predicted_{nutrient}_{exp_id}.png")
        plt.savefig(plot_filename)
        plt.close()  # Close figure to free memory

        print(f"✅ Scatter Plot saved: {plot_filename}")

# 🔹 Manually mapping experiment IDs to batch sizes
EXPERIMENT_BATCH_MAPPING = {
    "3555bffe": 64,  # Experiment ID -> Batch Size
    "c2f522fe": 128,
    "a9e0b6b4": 32,
}

def plot_batch_size_vs_convergence():
    """
    Generates a line chart showing validation loss over epochs for different batch sizes.
    Uses 'experiment_results_cheese_category_per_epoch.csv' as input.
    """
    # Load data
    file_path = os.path.join(LOGS_DIR, "experiment_results_cheese_category_per_epoch_batch.csv")
    if not os.path.exists(file_path):
        print(f"⚠️ File not found: {file_path}")
        return

    df = pd.read_csv(file_path)

    # 🔹 Map batch sizes using experiment IDs
    df["batch_size"] = df["experiment_id"].map(EXPERIMENT_BATCH_MAPPING)

    # Ensure data is sorted by batch size and epoch
    df = df.sort_values(by=["batch_size", "epoch"])

    # Get unique batch sizes
    batch_sizes = df["batch_size"].unique()

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot each batch size separately
    for batch_size in batch_sizes:
        subset = df[df["batch_size"] == batch_size]
        plt.plot(subset["epoch"], subset["val_loss"], marker="o", linestyle="-", label=f"Batch Size {batch_size}")

    # Labels and title
    plt.xlabel("Αριθμός εποχών (Epochs)")
    plt.ylabel("Validation Loss")
    plt.title("Σύγκριση Μεγέθους Batch και Σύγκλισης του Μοντέλου")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_filename = os.path.join(PLOTS_DIR, "batch_size_vs_convergence.png")
    plt.savefig(plot_filename, dpi=300)

    print(f"✅ Plot saved as {plot_filename}")


def plot_loss_function_comparison():
    """
    Generates a bar chart comparing different loss functions based on
    Train Loss, Validation Loss, and R² Score.
    """
    # Load data
    file_path = os.path.join(LOGS_DIR, "experiment_results_cheese_category_loss_function.csv")
    if not os.path.exists(file_path):
        print(f"⚠️ File not found: {file_path}")
        return

    df = pd.read_csv(file_path)

    # Set up the bar chart
    x_labels = df["loss_function"]
    x = range(len(x_labels))

    # Plot bars for each metric
    width = 0.2  # Bar width
    plt.figure(figsize=(10, 6))

    plt.bar(x, df["final_train_loss"], width=width, label="Train Loss", color="blue", alpha=0.6)
    plt.bar([i + width for i in x], df["best_validation_loss"], width=width, label="Validation Loss", color="red", alpha=0.6)
    plt.bar([i + 2 * width for i in x], df["r2_score"], width=width, label="R² Score", color="green", alpha=0.6)

    # Formatting
    plt.xticks([i + width for i in x], x_labels)  # Center labels
    plt.xlabel("Loss Function")
    plt.ylabel("Metric Value")
    plt.title("Comparison of Loss Functions")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Save plot
    plot_filename = os.path.join(PLOTS_DIR, "loss_function_comparison.png")
    plt.savefig(plot_filename, dpi=300)

    print(f"✅ Plot saved as {plot_filename}")


