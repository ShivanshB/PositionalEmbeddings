import os
import numpy as np
import glob
from transformers import AutoTokenizer
from tqdm import tqdm

def tokenize_and_save(input_dir, output_dir, tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    os.makedirs(output_dir, exist_ok=True)
    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
    
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
    output_dir = "/home/shiv/gutenberg-tokenized"
    tokenizer_name = "gpt2"
    
    for split in ['train', 'test', 'validation']:
        input_dir = os.path.join(base_dir, split)
        output_dir = os.path.join(os.path.join(output_dir, tokenizer_name), split)

        print(f"Processing {split} split...")
        tokenize_and_save(input_dir, output_dir, tokenizer_name)
        print(f"Finished processing {split} split.")

if __name__ == "__main__":
    main()