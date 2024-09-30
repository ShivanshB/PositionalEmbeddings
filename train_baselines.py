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
train_len = 128
val_lengths = [128, 256, 512]
test_lengths = [128, 256, 512]
n_hidden = 256 # reduced from 1024 for POC
n_depth = 8 # reduced from 16 for POC
n_heads = 8
num_epochs = 5 # reduced from 100 for POC
batch_size = 64 # reduced from 64 for POC
learning_rate = 3e-4
checkpoint_interval =  5
checkpoint_path = "checkpoints/experiment_1"
data_base_dir = "/home/shiv/gutenberg-tokenized-truncated/gpt2-processed"

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

class GutenbergDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data = np.load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids = self.data[idx]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
        }

# lambda to make creating datasets cleaner
g_path = lambda split, n_len: os.path.join(data_base_dir, split, f"data_{n_len}.npy")

print("loading training dataset")
# create training dataset
train_dataset = GutenbergDataset(g_path('train', train_len))

# create val datasets
print("loading validation datasets")
val_datasets = {val_len: GutenbergDataset(g_path('validation', val_len)) for val_len in val_lengths}

# create test datasets
print("loading test datasets")
test_datasets = {test_len: GutenbergDataset(g_path('test', test_len)) for test_len in test_lengths}

# training loop
def train_model(model, train_dataset, val_datasets, num_epochs, batch_size, learning_rate):
    model = AutoregressiveWrapper(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
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
        val_results = evaluate_model(model, val_datasets)
       
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Perplexity: {val_perplexity:.4f}")
        
        for length, metrics in val_results.items():
            print(f"Val Loss ({length}): {metrics['loss']:.4f}, Val Perplexity ({length}): {metrics['perplexity']:.4f}")

        # logging dictionary
        log_dict = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
        }

        for length, metrics in val_results.items():
            log_dict[f"val_loss_{length}"] = metrics['loss']
            log_dict[f"val_perplexity_{length}"] = metrics['perplexity']
        
        # log to wandb
        wandb.log(log_dict)

        # save checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
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

# model eval
def evaluate_model(model, datasets):
    if not isinstance(model, AutoregressiveWrapper):
        model = AutoregressiveWrapper(model)

    model.eval()
    results = {}
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for length, dataset in datasets.items():
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
            total_loss = 0
            total_tokens = 0
            
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                masks = batch['masks'].to(device)
                loss = model(input_ids, mask=masks)
                total_loss += loss.item()
                total_tokens += masks.sum().item()

            avg_loss = total_loss / total_tokens
            perplexity = np.exp(avg_loss)
            results[length] = {"loss": avg_loss, "perplexity": perplexity}

    return results

# train each model
for model in models:
    print(f"Training {model.name}")

    # initialize wandb
    wandb.init(project="positional-embeddings", name=model.name, config={
        "vocab_size": vocab_size,
        "max_seq_len": max_seq_len,
        "train_len": train_len,
        "val_lengths": val_lengths,
        "test_lengths": test_lengths,
        "n_hidden": n_hidden,
        "n_depth": n_depth,
        "n_heads": n_heads,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate
    })

    wandb.watch(model)  # Watch the model to log gradients and parameters
    trained_model = train_model(model, train_dataset, val_datasets, num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate)
    
    # evaluate the model on test datasets
    test_results = evaluate_model(trained_model, test_datasets)
    
    # print statistics and log final test results
    for length, metrics in test_results.items():
        print(f"{model.name} test loss ({length}): {metrics['loss']}")
        print(f"{model.name} test perplexity ({length}): {metrics['perplexity']}")
        
        wandb.log({
            f"test_loss_{length}": metrics['loss'],
            f"test_perplexity_{length}": metrics['perplexity']
        })

    wandb.finish()  # finish the run for this model