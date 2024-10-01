import os
import numpy as np
import glob
from transformers import AutoTokenizer
from tqdm import tqdm

# limit on number of .txt files to tokenize for sake of runtime/memory
# set to None to use whole dataset
file_percent_limit = .001 # using only one percent of the total dataset
min_files = 2 # minimum number of files

def tokenize_and_save(input_dir, output_dir, tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    os.makedirs(output_dir, exist_ok=True)
    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))

    if file_percent_limit:
        n_truncate = max(int(len(txt_files) * (file_percent_limit/100.0)), min_files)
        txt_files = txt_files[:n_truncate]
    
    for txt_file in tqdm(txt_files, desc=f"Tokenizing files in {input_dir}"):
        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read()
        tokens = tokenizer.encode(text)
        base_name = os.path.basename(txt_file)
        output_file = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}.npy")
        np.save(output_file, np.array(tokens, dtype=np.int32))

def main():
    # params to tokenize gutenberg dataset
    base_dir = "/home/shiv/gutenberg"
    output_dir_base = "/home/shiv/gutenberg-tokenized-truncated"
    tokenizer_name = "gpt2"
    
    for split in ['train', 'test', 'validation']:
        input_dir = os.path.join(base_dir, split)
        output_dir = os.path.join(output_dir_base, tokenizer_name, split)

        print(f"Processing {split} split...")
        tokenize_and_save(input_dir, output_dir, tokenizer_name)
        print(f"Finished processing {split} split.")

if __name__ == "__main__":
    main()