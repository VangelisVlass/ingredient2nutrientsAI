# 🚀 Food Nutrient Prediction with DistilBERT

This project predicts nutrient values based on food ingredients using **DistilBERT** and provides a scalable and configurable pipeline for food data processing and modeling.

---
## **📌 Key Features**
- **Ingredient-Based Nutrient Prediction:** Uses DistilBERT to predict nutrient values from ingredient lists.
- **Dynamic Experiment Configuration:** Centralized `CONFIG` dictionary controls all experimental settings.
- **Data Normalization Control:** Optionally apply **MinMax Scaling** via `normalize_data`.
- **Automated Logging:** Tracks dataset processing, training progress, and model performance.
- **Customizable Training Pipeline:** Modify loss functions, learning rates, and batch sizes easily.

---
## **📌 Setup Instructions**

### **1️⃣ Create and Activate Virtual Environment**

#### **🔹 Windows (CMD / PowerShell)**
```sh
python -m venv my_env  # Create environment
my_env\Scripts\activate  # Activate environment
```

#### **🔹 Mac/Linux**
```sh
python3 -m venv my_env  # Create environment
source my_env/bin/activate  # Activate environment
```

---
## **📌 Install Dependencies**

Inside the activated virtual environment, install the required libraries:

```sh
pip install -r requirements.txt
```

If you don’t have `requirements.txt`, create it:
```sh
touch requirements.txt
```
Then add the following:
```txt
transformers
torch
tqdm
pandas
numpy
dask
scikit-learn
joblib
```

---
## **📌 Data Preparation**

Before training, process and tokenize the dataset.

```sh
python main.py
```

This will:
1. Process the raw food data, sample it, and normalize if enabled.
2. Tokenize ingredients using DistilBERT's tokenizer.
3. Train the model and log results.

---
## **📌 Running the Model**

To start training with different learning rates:
```sh
python main.py
```
This script will log execution details in `logs/training_results.csv` and `logs/process_log.txt`.

---
## **📌 Logging System**
- **`logs/process_log.txt`** → Logs dataset processing steps.
- **`logs/training_results.csv`** → Stores experiment results, including loss, MAE, and R² scores.
- **`logs/validation_predictions_<EXPERIMENT_ID>.csv`** → Saves predicted vs actual values for validation.
- **Unique Experiment ID** ensures each run is traceable.

---
## **📌 Experiment Configuration**

All experiment settings are controlled in the `CONFIG` dictionary in `main.py`. Some key parameters:
```python
CONFIG = {
    "model_used": "distilbert-base-uncased",
    "learning_rates": [1.2e-4],  
    "batch_size": 64,
    "epochs": 12,
    "normalize_data": True,  # Enable MinMax Scaling
    "loss_function": "SmoothL1Loss",
    "nutrients_predicted": ["Protein", "Total Fat", "Sodium, Na", "Total Sugar", "Total Fiber"]
}
```

To modify an experiment, **update `CONFIG`** before running `main.py`.

---
## **📌 Authors**
👨‍💻 Created by: **Evangelos Stylianos Vlassopoulos** 

