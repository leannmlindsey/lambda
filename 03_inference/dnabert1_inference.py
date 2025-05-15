import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import gc
import psutil  # For monitoring CPU memory
import numpy as np
from datetime import datetime
import math
import argparse
from pathlib import Path

def add_seq_id(input_file):
    print(f"Processing {input_file}...")
    # Read the CSV
    df = pd.read_csv(input_file)

    # Add Seq_Id column as index starting from 1
    df['Seq_Id'] = range(1, len(df) + 1)

    # Save back to same file
    df.to_csv(input_file, index=False)
    print(f"Added Seq_Id column to {input_file}")

# More accurate time estimation
def estimate_remaining_time(total_sequences, processed_so_far, sequences_in_current_chunk,
                          chunk_time, start_time):
    CHUNK_SIZE = sequences_in_current_chunk
    # Calculate actual completion percentage
    total_remaining = total_sequences - processed_so_far - sequences_in_current_chunk
    chunks_remaining = math.ceil(total_remaining / CHUNK_SIZE)

    # Calculate average time per chunk based on elapsed time
    elapsed_time = time.time() - start_time
    chunks_processed = math.ceil((processed_so_far + sequences_in_current_chunk) / CHUNK_SIZE)
    avg_time_per_chunk = elapsed_time / max(chunks_processed, 1)  # Avoid division by zero

    # Estimate remaining time using moving average
    estimated_remaining_time = chunks_remaining * avg_time_per_chunk

    # Calculate actual progress percentage
    progress_percent = ((processed_so_far + sequences_in_current_chunk) / total_sequences) * 100

    return {
        'estimated_minutes': estimated_remaining_time / 60,
        'progress_percent': progress_percent,
        'chunks_remaining': chunks_remaining,
        'avg_chunk_time': avg_time_per_chunk
    }


def print_memory_usage():
    """Print both CPU and GPU memory usage"""
    # CPU Memory
    process = psutil.Process(os.getpid())
    print(f"CPU Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    # GPU Memory
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

class DNASequenceDataset(Dataset):
    def __init__(self, sequences, seq_ids, tokenizer, max_length=512):
        self.sequences = sequences
        self.seq_ids = seq_ids
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = str(self.sequences[idx])
        seq_id = self.seq_ids[idx]
        
        encoding = self.tokenizer(
            sequence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'seq_id': seq_id
        }

import time

def process_chunk(model, sequences, seq_ids, tokenizer, device, batch_size=8):
    """Process a single chunk of sequences"""
    chunk_start_time = time.time()
    chunk_dataset = DNASequenceDataset(sequences, seq_ids, tokenizer)
    chunk_dataloader = DataLoader(
        chunk_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )
    
    chunk_predictions = []
    chunk_seq_ids = []
    
    model.eval()
    with torch.no_grad():
        for batch in chunk_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_seq_ids = batch['seq_id']
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            batch_predictions = torch.sigmoid(logits).squeeze().cpu().numpy()
            
            chunk_predictions.extend(batch_predictions.tolist())
            chunk_seq_ids.extend(batch_seq_ids)
            
            # Clear GPU memory after each batch
            del input_ids, attention_mask, outputs, logits
            torch.cuda.empty_cache()
    
    # Clear memory
    del chunk_dataset, chunk_dataloader
    gc.collect()
    
    return chunk_seq_ids, chunk_predictions

def main(DATASET_PATH,OUTPUT_PATH,MODEL_CHECKPOINT):
    start_datetime = datetime.now()
    start_time = time.time()
    add_seq_id(DATASET_PATH)
    print(f"Processing input file: {DATASET_PATH}")
    print(f"Results will be saved to: {OUTPUT_PATH}")
    CHUNK_SIZE = 100  # Reduced chunk size
    BATCH_SIZE = 8
    
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available! This script requires GPU for processing.")
    
    DEVICE = torch.device("cuda")
    print(f"Using device: {DEVICE}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print_memory_usage()
    
    try:
        # Load model and tokenizer first
        print("Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_CHECKPOINT,
            local_files_only=True,
            trust_remote_code=True
        ).to(DEVICE)
        print("Model loaded")
        print_memory_usage()
        
        # Open output file for writing results incrementally
        with open(OUTPUT_PATH, 'w') as f:
            f.write('SeqID,prediction\n')  # Write header
        
        # Process the CSV file in chunks using pandas
        print("Processing dataset in chunks...")
        chunk_iterator = pd.read_csv(
            DATASET_PATH,
            chunksize=CHUNK_SIZE,  # Process this many rows at a time
            usecols=['Seq_Id', 'sequence']  # Only read the columns we need
        )
        
        total_processed = 0
        for chunk_num, chunk_df in enumerate(chunk_iterator):
            chunk_start = time.time()
            sequences_in_chunk = len(chunk_df)
            print(f"\nProcessing chunk {chunk_num + 1}")
            print(f"Rows {total_processed} to {total_processed + len(chunk_df)}")
            print_memory_usage()
            
            # Process chunk
            chunk_seq_ids, chunk_predictions = process_chunk(
                model,
                chunk_df['sequence'].values,
                chunk_df['Seq_Id'].values,
                tokenizer,
                DEVICE,
                BATCH_SIZE
            )
            
            # Write results for this chunk immediately
            with open(OUTPUT_PATH, 'a') as f:
                for sid, pred in zip(chunk_seq_ids, chunk_predictions):
                    f.write(f'{sid},{pred}\n')
            
            # Update progress
            total_processed += len(chunk_df)
            
            # Calculate and print timing information
            chunk_time = time.time() - chunk_start
            estimate = estimate_remaining_time(
                total_sequences=32922,
                processed_so_far=total_processed,
                sequences_in_current_chunk=sequences_in_chunk,
                chunk_time=chunk_time,
                start_time=start_time
            )
            
            print(f"Progress: {estimate['progress_percent']:.1f}%")
            print(f"Estimated remaining time: {estimate['estimated_minutes']:.1f} minutes")
            print(f"Average time per chunk: {estimate['avg_chunk_time']:.1f} seconds")
    
            total_processed += sequences_in_chunk
            
            # Clear memory
            del chunk_df, chunk_seq_ids, chunk_predictions
            gc.collect()
            print_memory_usage()
        
        print(f"\nProcessing complete! Results saved to {OUTPUT_PATH}")
        print(f"Total sequences processed: {total_processed}")
        end_datetime = datetime.now()
        total_time_taken = (end_datetime - start_datetime).total_seconds()
        print(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Actual time taken: {total_time_taken/60:.1f} minutes")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise
    finally:
        # Clean up
        print("\nFinal memory state:")
        print_memory_usage()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process DNA sequences using DNABERT2')
    parser.add_argument('input_file', type=str, help='Path to input CSV file')
    parser.add_argument('output_file', type=str, help='Path to save output CSV file')
    parser.add_argument('model_ckpt', type=str, help='Model checkpoint path')
    
    args = parser.parse_args()
    
    main(args.input_file, args.output_file, args.model_ckpt)
