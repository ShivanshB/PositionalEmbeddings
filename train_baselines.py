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
checkpoints = False # set to true before running main experiments
sigma = 0.1 # for stop positional embeddings
checkpoint_path = "checkpoints/experiment_1"
data_base_dir = "/home/shiv/gutenberg-tokenized-truncated/gpt2-processed"

class StochasticPositionalEmbedding(nn.Module): # currently implemented on top of learned positional embeddings
    def __init__(self, token_embedding, dim, max_seq_len=512, sigma=0.1):
        """
        Args:
            token_embedding (nn.Embedding): Shared token embedding table.
            dim (int): Embedding dimension.
            max_seq_len (int): Maximum sequence length.
            sigma (float): Standard deviation for the noise.
        """
        super().__init__()
        self.token_embedding = token_embedding  # Shared token embedding
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.sigma = sigma

        # Initialize learned positional embeddings
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.position_ids = torch.arange(max_seq_len).unsqueeze(0)  # Shape: (1, max_seq_len)

    def forward(self, batch_size):
        """
        Args:
            batch_size (int): Batch size.

        Returns:
            Tensor: Positional embeddings with stochastic noise. Shape: (batch_size, max_seq_len, dim)
        """
        # get device from token embedding table
        device = self.token_embedding.weight.device

        # Get standard learned positional embeddings
        pos_embeddings = self.pos_emb(self.position_ids.to(device))  # Shape: (1, max_seq_len, dim)
        pos_embeddings = pos_embeddings.repeat(batch_size, 1, 1)  # Shape: (batch_size, max_seq_len, dim)

        # Sample noise for each token
        # Token embedding matrix shape: (n_tokens, dim)
        # We need to sample noise per token and apply it to the embeddings
        with torch.no_grad():
            n_tokens, d = self.token_embedding.weight.size()
            noise = torch.randn(n_tokens, d, device=device) * self.sigma  # Shape: (n_tokens, dim)
            # Normalize noise per token if needed
            # noise = noise / noise.norm(dim=1, keepdim=True)

        # To apply noise based on the tokens in the batch, we need access to the input tokens
        # However, positional embeddings are typically independent of the input tokens.
        # Instead, we can sample a new noise vector for each position in each batch.

        # Sample noise for each position in the batch
        noise = torch.randn(batch_size, self.max_seq_len, self.dim, device=device) * self.sigma  # Shape: (batch_size, max_seq_len, dim)

        # Add noise to positional embeddings
        pos_embeddings = pos_embeddings + noise

        return pos_embeddings

# creating named transformer wrapper object for convenience
class TransformerWrapperNamed(TransformerWrapper):
    def __init__(self, name, pos_emb_type, token_embedding, custom_pos_emb=None, *args, **kwargs):
        """
        Args:
            name (str): Name of the positional embedding type.
            pos_emb_type (str): Type of positional embedding ('rotary', 'alibi', 'no_pos', 'learned', 'stochastic').
            token_embedding (nn.Embedding): Shared token embedding table.
            custom_pos_emb (nn.Module, optional): Custom positional embedding module.
        """
        super().__init__(*args, **kwargs)
        self.name = name
        self.pos_emb_type = pos_emb_type
        self.custom_pos_emb = custom_pos_emb

    def forward(self, tokens, **kwargs):
        """
        Args:
            tokens (Tensor): Input token indices of shape (batch_size, seq_len)
            **kwargs: Additional arguments.

        Returns:
            Tensor: Output logits of shape (batch_size, seq_len, num_tokens)
        """
        batch_size, seq_len = tokens.size()

        # Token Embeddings
        token_embeddings = self.token_emb(tokens)  # Shape: (batch_size, seq_len, embed_dim)

        # Positional Embeddings
        if self.pos_emb_type == 'stochastic' and self.custom_pos_emb is not None:
            pos_embeddings = self.custom_pos_emb(batch_size)[:, :seq_len, :]  # Shape: (batch_size, seq_len, embed_dim)
        elif self.use_abs_pos_emb:
            positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, seq_len)
            pos_embeddings = self.pos_emb(positions)[:, :seq_len, :]
        else:
            pos_embeddings = 0  # No positional embeddings

        # Combine embeddings
        x = token_embeddings + pos_embeddings  # Shape: (batch_size, seq_len, embed_dim)

        # Pass through transformer layers
        x = self.transformer(x, **kwargs)  # Shape: (batch_size, seq_len, embed_dim)



def create_model(pos_emb_type, sigma=0.1):
    """
    Creates a Transformer model with the specified positional embedding type.

    Args:
        pos_emb_type (str): Type of positional embedding ('rotary', 'alibi', 'no_pos', 'learned', 'stochastic').
        sigma (float): Standard deviation for stochastic positional embedding.

    Returns:
        TransformerWrapperNamed: Configured Transformer model.
    """
    # Initialize a shared token embedding table
    token_embedding = nn.Embedding(vocab_size, n_hidden)

    # Initialize custom positional embedding if needed
    custom_pos_emb = None
    if pos_emb_type == 'stop':
        custom_pos_emb = StochasticPositionalEmbedding(
            token_embedding=token_embedding,
            dim=n_hidden,
            max_seq_len=max_seq_len,
            sigma=sigma
        )

    # Initialize the TransformerWrapperNamed with the shared token embedding
    model = TransformerWrapperNamed(
        name=pos_emb_type,
        pos_emb_type=pos_emb_type,
        token_embedding=token_embedding,
        custom_pos_emb=custom_pos_emb,
        num_tokens=vocab_size,
        max_seq_len=max_seq_len,
        use_abs_pos_emb=(pos_emb_type == 'learned'),  # Disable absolute positional embeddings if 'learned' or 'stochastic'
        attn_layers=Decoder(
            dim=n_hidden,
            depth=n_depth,
            heads=n_heads,
            rotary_pos_emb=(pos_emb_type == 'rotary'),
            alibi_pos_bias=(pos_emb_type == 'alibi'),
            disable_abs_pos_emb=True  # disable absolute positional embeddings within attention layers
        )
    )

    # Share the token embedding table with the model's token embeddings
    model.token_emb = token_embedding  # Assign the shared embedding

    return model    

# Create models
models = [
    create_model('rotary'),
    create_model('alibi'),
    create_model('no_pos'),
    create_model('learned'),
    create_model('stop', sigma=sigma)
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
       
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")
        
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
        if checkpoints and (epoch + 1) % checkpoint_interval == 0:
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
        if checkpoints and val_results[train_len]["loss"] < best_val_loss:
            best_val_loss = val_results[train_len]["loss"]
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
            
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                loss = model(input_ids)
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
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