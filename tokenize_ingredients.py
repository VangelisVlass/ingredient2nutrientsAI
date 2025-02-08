from transformers import DistilBertTokenizer
import pandas as pd
from tqdm import tqdm 
from multiprocessing import Pool, cpu_count 
from log_manager import log_message
import os  

# Initialize tokenizer globally to avoid reloading inside processes
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

BATCH_SIZE = 256  # Number of rows to process in each batch
PARTIAL_SAVE_INTERVAL = 1000  # Save every 1000 processed rows

def tokenize_batch(ingredients_list , max_length=128):
    """
    Tokenize a batch of ingredient texts using DistilBERT.
    Returns tokenized input_ids and attention_masks.
    """
    try:
        tokenized = tokenizer(
            ingredients_list.tolist(),
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True
        )
        return {"input_ids": tokenized["input_ids"], "attention_mask": tokenized["attention_mask"]}
    except Exception as e:
        log_message(f"🚨 Tokenization error: {e}")
        return {"input_ids": [], "attention_mask": []} 

def tokenize_ingredients(input_file="data.csv", output_file="tokenized_data.csv", long_file="long_ingredients.csv", max_length = 128):
    """
    Tokenizes the ingredients column and saves valid numerical lists instead of numpy objects.
    Ingredients that exceed MAX_LENGTH are removed and stored separately.
    """
    log_message("🔄 Loading data for tokenization...")
    
    if not os.path.exists(input_file):
        log_message(f"🚨 Error: File '{input_file}' not found!")
        return

    data = pd.read_csv(input_file)
    
    if "ingredients" not in data.columns:
        log_message("🚨 Error: 'ingredients' column not found in the dataset!")
        return

    log_message("🔄 Filtering long ingredient lists...")
    token_length_check = data["ingredients"].astype(str).apply(lambda x: len(tokenizer.tokenize(x)))
    
    long_ingredients = data[token_length_check > max_length]
    short_ingredients = data[token_length_check <= max_length]

    if not long_ingredients.empty:
        long_dir = os.path.dirname(long_file)
        if long_dir:  
            os.makedirs(long_dir, exist_ok=True)  
        long_ingredients.to_csv(long_file, index=False, encoding="utf-8")
        log_message(f"🚨 Saved {len(long_ingredients)} long ingredient entries to '{long_file}'.")

    log_message(f"🔹 Keeping {len(short_ingredients)} rows for tokenization.")

    batches = [short_ingredients["ingredients"][i:i+BATCH_SIZE] for i in range(0, len(short_ingredients), BATCH_SIZE)]

    log_message(f"🔄 Tokenizing in {len(batches)} batches using {min(4, cpu_count())} CPU cores...")

    with Pool(min(4, cpu_count())) as pool:
        tokenized_batches = list(tqdm(pool.imap(tokenize_batch, batches), total=len(batches), desc="🔄 Tokenizing"))

    log_message("✅ Merging tokenized results...")
    
    input_ids_list = []
    attention_mask_list = []
    
    for i, batch in enumerate(tokenized_batches):
        input_ids_list.extend(batch["input_ids"])
        attention_mask_list.extend(batch["attention_mask"])
        
        if len(input_ids_list) % PARTIAL_SAVE_INTERVAL == 0:
            temp_df = pd.DataFrame({"input_ids": input_ids_list, "attention_mask": attention_mask_list})
            temp_df.to_csv(f"tokenized_partial_{len(input_ids_list)}.csv", index=False, encoding="utf-8")
            log_message(f"✅ Saved {len(input_ids_list)} tokenized rows so far...")

    log_message("✅ Saving final tokenized dataset...")
    short_ingredients = short_ingredients.copy()
    short_ingredients.loc[:, "input_ids"] = input_ids_list
    short_ingredients.loc[:, "attention_mask"] = attention_mask_list
    short_ingredients.to_csv(output_file, index=False, encoding="utf-8")

    log_message(f"✅ Tokenization complete. Saved to '{output_file}'.")