"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from dataset import Multi30kDataset
from lr_scheduler import NoamScheduler
from typing import Optional
from model import Transformer, make_src_mask, make_tgt_mask


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS  
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing


    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        # TODO: Task 3.1
        dist = torch.full_like(logits, self.smoothing / (self.vocab_size - 1))
        dist.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
        dist[:, self.pad_idx] = 0
        dist.masked_fill_((target == self.pad_idx).unsqueeze(1), 0)
        loss = torch.mean(torch.sum(-dist * torch.log_softmax(logits, dim=1), dim=1))
        return loss

        


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP  
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).

    """
    
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    num_batches = 0

    with torch.set_grad_enabled(is_train):

        for src, tgt in data_iter:
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_in = tgt[:, :-1]

            src_mask = make_src_mask(src)
            tgt_mask = make_tgt_mask(tgt_in)

            logits = model(src, tgt_in, src_mask, tgt_mask)
            logits_flat = logits.view(-1, logits.size(-1))

            tgt_flat = tgt[:, 1:].contiguous().view(-1)

            loss = loss_fn(logits_flat, tgt_flat)
            total_loss += loss.item()
            num_batches += 1

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            
    avg_loss = total_loss / num_batches
    return avg_loss






# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING  
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.

    """
    # TODO: Task 3.3 — implement token-by-token greedy decoding
    with torch.no_grad():
        model.eval()
        memory = model.encode(src, src_mask)
        ys = torch.tensor([[start_symbol]], device=device)

        with torch.no_grad():
            for _ in range(max_len - 1):
                tgt_mask = make_tgt_mask(ys)
                decoded = model.decode(memory, src_mask, ys, tgt_mask)
                next_token = decoded[:, -1, :].argmax(dim=-1).item()
                ys = torch.cat([ys, torch.tensor([[next_token]], device=device)], dim=1)  

                if next_token == end_symbol:
                    break

    return ys  


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION  
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
                          Each batch yields (src, tgt) token-index tensors.
        tgt_vocab       : Vocabulary object with idx_to_token mapping.
                          Must support  tgt_vocab.itos[idx]  or
                          tgt_vocab.lookup_token(idx).
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).

    """
    # TODO: Task 3 — loop test set, decode, compute and return BLEU
    from bleu import list_bleu
    model.eval()

    hypotheses = []
    references = []

    sos_idx = tgt_vocab['<sos>']
    eos_idx = tgt_vocab['<eos>']
    pad_idx = tgt_vocab['<pad>']

    idx2tok = {idx: tok for tok, idx in tgt_vocab.items()}

    with torch.no_grad():
        for src, tgt in test_dataloader:

            for i in range(src.size(0)):
                src_i = src[i].unsqueeze(0).to(device)
                src_mask = make_src_mask(src_i)
                ys = greedy_decode(model, src_i, src_mask, max_len, sos_idx, eos_idx, device)
                tok = ys.squeeze().tolist()
                if eos_idx in tok:
                    tok = tok[:tok.index(eos_idx)]
                hyp = " ".join(idx2tok[idx] for idx in tok if idx != pad_idx)
                hyp = hyp.replace(" .", ".").replace(" ,", ",").replace(" !", "!").replace(" ?", "?").replace(" '", "'")
                hypotheses.append(hyp.lower())

                
                ref = tgt[i].tolist()
                if eos_idx in ref:
                    ref = ref[1:ref.index(eos_idx)]
                else:
                    ref = ref[1:]

                ref = " ".join(idx2tok[idx] for idx in ref if idx != pad_idx)
                ref = ref.replace(" .", ".").replace(" ,", ",").replace(" !", "!").replace(" ?", "?").replace(" '", "'")
                references.append(ref.lower())  

    bleu_score = list_bleu([references], hypotheses)
    return bleu_score


# ══════════════════════════════════════════════════════════════════════
# ❺  CHECKPOINT UTILITIES  (autograder loads your model from disk)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    The autograder will call load_checkpoint to restore your model.
    Do NOT change the keys in the saved dict.

    Args:
        model     : Transformer instance.
        optimizer : Optimizer instance.
        scheduler : NoamScheduler instance.
        epoch     : Current epoch number.
        path      : File path to save to (default 'checkpoint.pt').

    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'

    model_config must contain all kwargs needed to reconstruct
    Transformer(**model_config), e.g.:
        {'src_vocab_size': ..., 'tgt_vocab_size': ...,
         'd_model': ..., 'N': ..., 'num_heads': ...,
         'd_ff': ..., 'dropout': ...}
    """
    # TODO: implement using torch.save({...}, path)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'model_config': {
            'src_vocab_size': model.src_vocab_size,
            'tgt_vocab_size': model.tgt_vocab_size,
            'd_model': model.d_model,
            'N': model.N,
            'num_heads': model.num_heads,
            'd_ff': model.d_ff,
            'dropout': model.p,
            'de_vocab': model.de_vocab,
            'en_vocab': model.en_vocab
        }
    }
    torch.save(checkpoint, path)



def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Args:
        path      : Path to checkpoint file saved by save_checkpoint.
        model     : Uninitialised Transformer with matching architecture.
        optimizer : Optimizer to restore (pass None to skip).
        scheduler : Scheduler to restore (pass None to skip).

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).

    """
    # TODO: implement restore logic
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch']


# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in src_batch], batch_first=True, padding_value=1)
    tgt_batch = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in tgt_batch], batch_first=True, padding_value=1)
    return src_batch, tgt_batch

def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.

    Steps:
        1. Init W&B:   wandb.init(project="da6401-a3", config={...})
        2. Build dataset / vocabs from dataset.py
        3. Create DataLoaders for train / val splits
        4. Instantiate Transformer with hyperparameters from config
        5. Instantiate Adam optimizer (β1=0.9, β2=0.98, ε=1e-9)
        6. Instantiate NoamScheduler(optimizer, d_model, warmup_steps=4000)
        7. Instantiate LabelSmoothingLoss(vocab_size, pad_idx, smoothing=0.1)
        8. Training loop:
               for epoch in range(num_epochs):
                   run_epoch(train_loader, model, loss_fn,
                             optimizer, scheduler, epoch, is_train=True)
                   run_epoch(val_loader, model, loss_fn,
                             None, None, epoch, is_train=False)
                   save_checkpoint(model, optimizer, scheduler, epoch)
        9. Final BLEU on test set:
               bleu = evaluate_bleu(model, test_loader, tgt_vocab)
               wandb.log({'test_bleu': bleu})
    """
    # TODO: implement full experiment

    print("Running training experiment... \n")

    print("Loading datasets and building vocabularies... \n")
    train_ds = Multi30kDataset(split='train')
    train_ds.build_vocab()
    train_ds.process_data()

    val_ds = Multi30kDataset(split='validation')
    val_ds.en_vocab = train_ds.en_vocab
    val_ds.de_vocab = train_ds.de_vocab
    val_ds.en_vocab_rev = train_ds.en_vocab_rev
    val_ds.de_vocab_rev = train_ds.de_vocab_rev
    val_ds.process_data()

    test_ds = Multi30kDataset(split='test')
    test_ds.en_vocab = train_ds.en_vocab
    test_ds.de_vocab = train_ds.de_vocab
    test_ds.en_vocab_rev = train_ds.en_vocab_rev
    test_ds.de_vocab_rev = train_ds.de_vocab_rev
    test_ds.process_data()



    print("Loaded datasets and built vocabularies. \n")


    wandb.init(project="da6401-a3", config={
        'src_vocab_size': len(train_ds.de_vocab),
        'tgt_vocab_size': len(train_ds.en_vocab),
        'd_model': 512,
        'N': 6,
        'num_heads': 8,
        'd_ff': 2048,
        'dropout': 0.1,
        'batch_size': 64,
        'num_epochs': 10
    })


    train_loader = DataLoader(train_ds, batch_size=wandb.config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=wandb.config.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=wandb.config.batch_size, collate_fn=collate_fn)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(F"Using device: {device} Initializing model and other components... \n")
    model = Transformer(
        src_vocab_size=wandb.config.src_vocab_size,
        tgt_vocab_size=wandb.config.tgt_vocab_size,
        d_model=wandb.config.d_model,
        N=wandb.config.N,
        num_heads=wandb.config.num_heads,
        d_ff=wandb.config.d_ff,
        dropout=wandb.config.dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9, lr=1.0)
    scheduler = NoamScheduler(optimizer, d_model=wandb.config.d_model, warmup_steps=4000)
    loss_fn = LabelSmoothingLoss(vocab_size=wandb.config.tgt_vocab_size, pad_idx=train_ds.en_vocab['<pad>'], smoothing=0.1)

    model.de_vocab = train_ds.de_vocab
    model.en_vocab = train_ds.en_vocab

    print(" Intialization done. Starting training loop... \n")

    for epoch in range(wandb.config.num_epochs):
        train_loss = run_epoch(train_loader, model, loss_fn, optimizer, scheduler, epoch, is_train=True, device=device)
        val_loss = run_epoch(val_loader, model, loss_fn, None, None, epoch, is_train=False, device=device)        
        wandb.log({'train_loss': train_loss, 'val_loss': val_loss}, step=epoch)
        save_checkpoint(model, optimizer, scheduler, epoch)

        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/{wandb.config.num_epochs} Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    blue_score = evaluate_bleu(model, test_loader, train_ds.en_vocab, device=device)
    wandb.log({'test_bleu': blue_score})
    print(f"Test BLEU: {blue_score:.2f}")

if __name__ == "__main__":
    run_training_experiment()
