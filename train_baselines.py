import os
import numpy as np
import torch
import tqdm
import wandb

from tqdm import tqdm
from x_transformers import TransformerWrapper, Decoder
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from x_transformers import AutoregressiveWrapper

# hparams
vocab_size = 50257 # vocab size corresponding to gpt2 tokenizer
max_seq_len = 512
max_train_len = 512
max_len_val = 2048
n_hidden = 1024
n_depth = 16
n_heads = 8
num_epochs = 100
batch_size = 64
learning_rate = 3e-4
checkpoint_interval =  5
checkpoint_path = "checkpoints/experiment_1"

# creating named transformer wrapper object for convenience
class TransformerWrapperNamed(TransformerWrapper):
    def __init__(self, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name

# helper function to create model types
def create_model(pos_emb_type):
    return TransformerWrapperNamed(
        name=pos_emb_type,
        num_tokens=vocab_size,
        max_seq_len=max_seq_len,
        use_abs_pos_emb=(pos_emb_type == 'learned'),  # disable absolute positional embeddings, 
        attn_layers=Decoder(
            dim=n_hidden,
            depth=n_depth,
            heads=n_heads,
            rotary_pos_emb=(pos_emb_type == 'rotary'),
            alibi_pos_bias=(pos_emb_type == 'alibi'),
            disable_abs_pos_emb=(pos_emb_type == 'learned') # disable absolute positional embeddings in attention layers
        )
    )

# Create models
models = [
    create_model('rotary'),
    create_model('alibi'),
    create_model('no_pos'),
    create_model('learned')
]

# tokenizing function
def tokenize_function(examples, max_length=max_train_len):
    return tokenizer(examples["text"], truncation=True, max_length=max_length)

# loading wikitext
wikitext_dataset = load_dataset("wikitext", "wikitext-2-v1")

# remove empty samples
filtered_dataset = wikitext_dataset.filter(lambda sample: len(sample['text'].strip()) > 0)

# load gpt2 tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# tokenize samples
tokenized_wikitext = filtered_dataset.map(tokenize_function, batched=True, remove_columns=["text"])


# dataset class
class WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset, split='train', max_length=max_seq_len):
        self.tokenized_dataset = tokenized_dataset[split]
        self.max_length = max_length

    def __len__(self):
        return len(self.tokenized_dataset)

    def __getitem__(self, idx):
        item = self.tokenized_dataset[idx]
        input_ids = item['input_ids'][:self.max_length]  # truncate if too long
        input_ids = input_ids + [0] * (self.max_length - len(input_ids))  # pad if too short
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long)
        }

        #TODO val on longer seq + wandb

# create the dataset
wikitext_dataset = WikiTextDataset(tokenized_wikitext, split='train')

# training loop
def train_model(model, train_dataset, val_dataset, num_epochs, batch_size, learning_rate):
    model = AutoregressiveWrapper(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            
            optimizer.zero_grad()
            loss = model(input_ids)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_dataloader)

        # validation
        avg_val_loss, val_perplexity = evaluate_model(model, val_dataset) 
       
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Perplexity: {val_perplexity:.4f}")
        
        # log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_perplexity": val_perplexity
        })

        # save checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }
            checkpoint_path = f"checkpoints/{model.net.name}_epoch_{epoch+1}.pt"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(checkpoint, checkpoint_path)
            wandb.save(checkpoint_path)  # Save to wandb as well

        # save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = f"checkpoints/{model.net.name}_best.pt"
            torch.save(model.state_dict(), best_model_path)
            wandb.save(best_model_path)

    return model


# create train and validation datasets
train_dataset = WikiTextDataset(tokenized_wikitext, split='train')
val_dataset = WikiTextDataset(tokenized_wikitext, split='validation')

# model eval
def evaluate_model(model, dataset):
    # wrap if not wrapped
    if not isinstance(model, AutoregressiveWrapper):
        model = AutoregressiveWrapper(model)

    model.eval()
    total_loss = 0
    total_tokens = 0
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            loss = model(input_ids)
            total_loss += loss.item()
            total_tokens += input_ids.numel()

    # val statistic calculations
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    return avg_loss, perplexity

# train each model on wikitext
for model in models:
    print(f"Training {model.name} on WikiText-103")

    # initialize wandb
    wandb.init(project="positional-embeddings", name=model.name, config={
        "vocab_size": vocab_size,
        "max_seq_len": max_seq_len,
        "max_train_len": max_train_len,
        "n_hidden": n_hidden,
        "n_depth": n_depth,
        "n_heads": n_heads,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate
    })

    wandb.watch(model)  # Watch the model to log gradients and parameters
    trained_model = train_model(model, train_dataset, val_dataset, num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate)
    
    # evaluate the model
    test_dataset = WikiTextDataset(tokenized_wikitext, split='test')
    test_loss, test_perplexity = evaluate_model(trained_model, test_dataset)

    # print statistics
    print(f"{model.name} test loss: {test_loss}")
    print(f"{model.name} perplexity: {test_perplexity}")
    
    # log final test loss
    wandb.log({
        "test_loss": test_loss,
        "test_perplexity": test_perplexity
    })

    wandb.finish()  # finish the run for this model