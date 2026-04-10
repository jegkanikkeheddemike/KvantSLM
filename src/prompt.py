"""
Inference / prompting script (Tiny Shakespeare, char-level).
Students will integrate sustainability tracking themselves.

Source: https://github.com/karpathy/nanoGPT
"""

import os
import pickle
from codecarbon import EmissionsTracker
import torch

from model import GPT, GPTConfig

# ----------------------------
# Edit these
# ----------------------------
OUT_DIR = "out"
CKPT_PATH = os.path.join(OUT_DIR, "ckpt.pt")

PROMPT = "To be, or not to be"
MAX_NEW_TOKENS = 200
TEMPERATURE = 1.0
TOP_K = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------


def load_meta(data_dir: str):
    meta_path = os.path.join(data_dir, "meta.pkl")
    with open(meta_path, "rb") as f:
        return pickle.load(f)


def main(max_tokens: int = MAX_NEW_TOKENS):
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    # train.py should store config with model parameters and data_dir
    data_dir = ckpt["config"]["data_dir"]
    model_cfg = ckpt["config"]["model"]

    meta = load_meta(data_dir)
    stoi = meta["stoi"]         # char to index mapping
    itos = meta["itos"]         # index to char mapping

    def encode(s: str):
        # map unknown chars to a safe fallback if needed
        return [stoi.get(ch, stoi[" "]) for ch in s]

    def decode(tokens):
        return "".join([itos[t] for t in tokens])

    config = GPTConfig(**model_cfg)
    model = GPT(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    idx = torch.tensor([encode(PROMPT)], dtype=torch.long, device=DEVICE)

    out = model.generate(
        idx,
        max_new_tokens=max_tokens,
        temperature=TEMPERATURE,
        top_k=TOP_K
    )

    print(decode(out[0].tolist()))

API_KEY = "cpt_fIKPwV3K982M8bP31Gi85XIgaO9GL9T8GaYlQX4udd0"
experiment_config = {
    "basePrompt-200" : "b05011ff-1cb7-4000-a3cf-93d4fc972da7",
    "AltPrompt1-100" : "217567db-1bce-425c-b690-41983b283e96",
    "AltPrompt1-200" : "98657273-a383-46fe-befa-4462a59b1977",
    "AltPrompt1-300" : "e2e25115-2c6b-40dc-b08e-8cea6781dbbb",
    "AltPrompt1-500" : "30bf7690-ca63-4916-9bc2-7abc8d528fe9",
    "AltPrompt1-700" : "86868eee-3c75-42b4-a238-d071ffdd0339",
    "AltPrompt1-1000" : "5501b958-7e61-4853-bf82-7afb4c628742",
    "AltPrompt2-4.4.128" : "4fe3a116-2dbf-4a73-a8ba-9bcca4fa8181",
    "AltPrompt2-8.4.128" : "b3b23a4a-23f9-49ee-b005-a973d60d71a2",
    "AltPrompt2-16.4.128" : "00b7c136-65e3-4c97-bfa3-e6e90035ce29",
    "AltPrompt2-4.8.128": "54a71f7d-e815-4ea4-9b90-f1abb10d43cd",
    "AltPrompt2-8.8.128": "a8524d76-0545-4ee5-bb92-92e27f6f442b",
    "AltPrompt2-16.8.128": "41bc8a04-803f-42f4-9a4b-54d26105ceb3",
    "AltPrompt2-4.16.128": "09c5f47e-2c12-428a-a9e0-633222e03bf6",
    "AltPrompt2-8.16.128": "e9938e84-48fd-4414-b886-ea4ccb288cac",
    "AltPrompt2-16.16.128": "ebe7d42c-f69d-42a9-9a71-0c319b08805b"
}

if __name__ == "__main__":
    for token_len in range(100, 1100, 100):
        print(f"___ Running experiment with max_tokens={token_len} ___")
        experiment_id = experiment_config[str(token_len)]
        for _ in range(10):
            tracker = EmissionsTracker(api_key=API_KEY, experiment_id=experiment_id, save_to_api=True, save_to_file=False)
            tracker.start()
            main()
            emissions = tracker.stop()
