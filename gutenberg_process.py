import os
import numpy as np
import glob
from tqdm import tqdm

def chunk_tokenized_sequence(tokenized_sequence, max_len):
    num_chunks = len(tokenized_sequence) // max_len
    if num_chunks > 0:
        token_chunks = np.array_split(tokenized_sequence[:num_chunks * max_len], num_chunks)
        return token_chunks
    return []

def process_dataset(dataset_folder, max_len):
    processed_data = []
    
    npy_files = glob.glob(os.path.join(dataset_folder, '*.npy'))
    
    for file_path in tqdm(npy_files, desc=f"Processing {dataset_folder}"):
        tokenized_sequence = np.load(file_path)
        chunks = chunk_tokenized_sequence(tokenized_sequence, max_len)
        processed_data.extend(chunks)

    return np.array(processed_data)

def save_processed_data(processed_data, output_folder, max_len):
    os.makedirs(output_folder, exist_ok=True)
    np.save(os.path.join(output_folder, f'data_{max_len}.npy'), processed_data)

def process_all_datasets(base_folder, model):
    datasets = ['train', 'test', 'validation']
    max_lens = [128, 256, 512]
    
    for dataset in datasets:
        dataset_folder = os.path.join(base_folder, model, dataset)
        
        for max_len in max_lens:
            # output folder path
            output_folder = os.path.join(base_folder, f"{model}-processed", dataset)

            # make if doesn't exist
            os.makedirs(output_folder, exist_ok=True)
            processed_data = process_dataset(dataset_folder, max_len)
            save_processed_data(processed_data, output_folder, max_len)

# params for reading and writing
base_folder = '/home/shiv/gutenberg-tokenized-truncated'
model = 'gpt2'
process_all_datasets(base_folder, model)
