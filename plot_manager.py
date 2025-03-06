import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

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
    sns.heatmap(df, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Correlation Matrix – Extracted Features vs. Nutrients")

    plot_filename = os.path.join(PLOTS_DIR, f"correlation_matrix_{exp_id}.png")
    plt.savefig(plot_filename)
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


### 🚀 **Run All Plots for a Given Experiment**
def generate_all_plots(exp_id):
    print(f"\n📊 Generating plots for experiment: {exp_id}\n")
    plot_feature_importance(exp_id)
    plot_loss_per_epoch(exp_id)
    plot_correlation_matrix(exp_id)
    plot_misclassification(exp_id)
    plot_predicted_vs_actual(exp_id)
    plot_nutrient_accuracy(exp_id)
    print(f"✅ All plots saved in {PLOTS_DIR}/")
