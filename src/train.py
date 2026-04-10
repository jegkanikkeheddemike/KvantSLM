"""
Course training script (simplified from nanoGPT).

Focus:
- Train a small GPT-style model from scratch on a tiny dataset.
- Students will integrate sustainability tracking themselves.

Source: https://github.com/karpathy/nanoGPT
"""

import os
import time
import pickle
from dataclasses import asdict

import numpy as np
import torch

from model import GPTConfig, GPT
from codecarbon import EmissionsTracker

# -----------------------------------------------------------------------------
# Experiment configuration

# I/O
OUT_DIR = "out"
DATA_DIR = os.path.join("data")
EVAL_INTERVAL = 200     
EVAL_ITERS = 50
LOG_INTERVAL = 50
SAVE_CHECKPOINT = True

# Model (main tunables)
N_LAYER = 4
N_HEAD = 4
N_EMBD = 128
DROPOUT = 0.1
BIAS = True

# Training (main parameters you can also experiment with)
SEED = 1
DEVICE = "cpu"          # If you can, try also seeing consumption when using gpu (change this to 'cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = "float32"       
BATCH_SIZE = 32         # Number of sequences processed in parallel.
BLOCK_SIZE = 256        # Maximum context length for predictions (e.g. 128 or 256). The longer the block size, the more memory and compute it requires, but it can also lead to better performance.
MAX_ITERS = 2000        # Total number of training iterations. The more iterations, the better the model can perform, but it also takes more time and energy to train.
LEARNING_RATE = 3e-4    # the standard starting learning rate, often good enough for a first try
WEIGHT_DECAY = 0.1      # L2 Regularization
GRAD_CLIP = 1.0         # To prevent exploding gradients

# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

def load_meta(data_dir: str):
    meta_path = os.path.join(data_dir, "meta.pkl")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "rb") as f:
        return pickle.load(f)

def get_batch(split: str, data_dir: str, block_size: int, batch_size: int, device: str):
    # simple, robust memmap loader
    bin_path = os.path.join(data_dir, f"{split}.bin")
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")

    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])

    x = x.to(device)
    y = y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model: GPT, data_dir: str, block_size: int, batch_size: int, device: str, eval_iters: int):
    model.eval()
    losses = {}
    for split in ["train", "val"]:
        split_losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            x, y = get_batch(split, data_dir, block_size, batch_size, device)
            _, loss = model(x, y)
            split_losses[k] = loss
        losses[split] = split_losses.mean().item()
    model.train()
    return losses

def save_checkpoint(out_dir: str, model: GPT, optimizer: torch.optim.Optimizer, iter_num: int, config: dict):
    os.makedirs(out_dir, exist_ok=True)
    ckpt = {
        "iter_num": iter_num,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "config": config,
    }
    torch.save(ckpt, os.path.join(out_dir, "ckpt.pt"))

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_seed(SEED)

    meta = load_meta(DATA_DIR)
    vocab_size = meta["vocab_size"] if meta and "vocab_size" in meta else 50304

    cfg = GPTConfig(
        block_size=BLOCK_SIZE,
        vocab_size=vocab_size,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        dropout=DROPOUT,
        bias=BIAS,
    )

    # create the model and move it to the device
    model = GPT(cfg).to(DEVICE)

    # create the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    # (optional) uncomment this for printing model size once
    # print(f"Device: {DEVICE}")
    # print(f"Model parameters: {model.get_num_params():,}")
    # print(f"Training for {MAX_ITERS} iterations | batch={BATCH_SIZE} | block={BLOCK_SIZE}")

    t0 = time.time()
    for it in range(MAX_ITERS + 1):
        # periodic evaluation
        if it % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, DATA_DIR, BLOCK_SIZE, BATCH_SIZE, DEVICE, EVAL_ITERS)
            dt = time.time() - t0
            print(f"iter {it:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | elapsed {dt:.1f}s")

            if SAVE_CHECKPOINT and it > 0:
                config_dump = {
                    "data_dir": DATA_DIR,
                    "train": {
                        "batch_size": BATCH_SIZE,
                        "block_size": BLOCK_SIZE,
                        "max_iters": MAX_ITERS,
                        "learning_rate": LEARNING_RATE,
                        "weight_decay": WEIGHT_DECAY,
                        "grad_clip": GRAD_CLIP,
                        "dtype": DTYPE,
                        "device": DEVICE,
                    },
                    "model": asdict(cfg),
                }
                save_checkpoint(OUT_DIR, model, optimizer, it, config_dump)

        # training step
        x, y = get_batch("train", DATA_DIR, BLOCK_SIZE, BATCH_SIZE, DEVICE)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if GRAD_CLIP and GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()

        if it % LOG_INTERVAL == 0:
            print(f"iter {it:5d} | loss {loss.item():.4f}")

    print("Training completed.")

    # Save final checkpoint
    if SAVE_CHECKPOINT:
        config_dump = {
            "data_dir": DATA_DIR,
            "train": {
                "batch_size": BATCH_SIZE,
                "block_size": BLOCK_SIZE,
                "max_iters": MAX_ITERS,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "grad_clip": GRAD_CLIP,
                "dtype": DTYPE,
                "device": DEVICE,
            },
            "model": asdict(cfg),
        }
        save_checkpoint(OUT_DIR, model, optimizer, MAX_ITERS, config_dump)


experiment_config = {
    "AltTrain1-4.4.128": "6a1c0f28-cfba-40ed-a953-245c99387425",
    "AltTrain1-8.4.128": "49e57eaa-3ec8-43b0-bb9a-1151fb03d836",
    "AltTrain1-16.4.128": "8fa867f0-e1a4-40fb-ae87-05269adeacfa",
    "AltTrain1-4.8.128": "e9dac05e-72aa-49bc-b64e-c6cda8f410a3",
    "AltTrain1-8.8.128": "7b24d8be-4762-4cac-bd68-47b7fdda8c06",
    "AltTrain1-16.8.128": "a53c7fc6-9517-44f0-92f9-d7279a00406a",
    "AltTrain1-4.16.128": "77dc9529-3d70-4f54-81c9-f9b5caa5d208",
    "AltTrain1-8.16.128": "0d46d988-d729-4c8b-9d3a-d2798b30ea6f",
    "AltTrain1-16.16.128": "840ed840-7811-4ddb-944a-dda3805837a8",
    "AltTrain2-4.4.128": "6cdc6c93-aeb4-4f20-ae19-21c88984a035",
    "AltTrain2-8.4.128": "d17cddfa-ad27-497b-800d-e731b46153f4",
    "AltTrain2-16.4.128": "1d9193f3-0a2c-4855-bec8-97ad5024b5f4",
    "AltTrain2-4.8.128": "f9f366ba-c65b-45d8-9419-7ebaa16c7a84",
    "AltTrain2-8.8.128": "4c9641b5-2932-4b81-9f45-24ece2361fda",
    "AltTrain2-16.8.128": "e6f9c565-1bb3-4c9e-ad8e-0f88d28ca1ff",
    "AltTrain2-4.16.128": "5b982fdc-4a84-4e50-adc3-af4d0b8be4ef",
    "AltTrain2-8.16.128": "7478b7a4-60ff-4ff7-9a84-ea0e44593813",
    "AltTrain2-16.16.128": "ba046bf0-551f-4288-a601-986bcad8e6c8",
    "AltTrain3-100": "e9948491-7c11-4082-ac0b-9baa585b6562",
    "AltTrain3-200": "2a47fdf5-d234-4a93-bf7b-85a7e0e116a9",
    "AltTrain3-300": "8b5a6da4-fccb-47ac-be90-df43b03afe9a",
    "AltTrain3-400": "ad5c01e8-296e-41ff-b87b-500b6582760c",
    "AltTrain3-500": "36f7b846-ac88-434e-a007-bbf6e3ccc466",
    "AltTrain3-700": "f43c2e4b-66d3-4945-a582-350e19e5c23a",
    "AltTrain3-1000": "d68c19a9-fed9-44dc-8042-3b665d552116",
    "AltTrain3-1300": "356dd618-8ffd-4d06-a10b-3ed184944fc1",
    "AltTrain3-1500": "8c721192-0d16-4de6-8112-b77eb9708ea7",
    "AltTrain3-1700": "d500b0f3-14bc-479d-ae4b-d527bff7f16a",
    "AltTrain3-2000": "f05d5b59-ba42-4a1f-a55a-4b1432e4c81e",
    "AltTrain3-2500": "5894ad93-67d4-4d61-be0c-11d6d0674c7c",
    "AltTrain3-3000": "da5db1e0-6d96-41bc-8b9b-36125fd0fd0a"
}

if __name__ == "__main__":
    tracker = EmissionsTracker(api_key=os.environ["API_KEY"], experiment_id=experiment_config["BaseLine"], save_to_api=True, save_to_file=False)
    tracker.start()
    main()
    emissions = tracker.stop()
