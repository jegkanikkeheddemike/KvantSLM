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
    "100" : "TODO",
    "200" : "TODO",
    "300" : "TODO",
    "400" : "TODO",
    "500" : "TODO",
    "600" : "TODO",
    "700" : "TODO",
    "800" : "TODO",
    "900" : "TODO",
    "1000" : "TODO",
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
