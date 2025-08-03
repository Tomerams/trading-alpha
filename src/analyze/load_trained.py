import json, torch, joblib
from pathlib import Path
from models.model_utilities import get_model

MODEL_DIR = Path("src/files/models")

def load_pipeline(ticker: str, target: str):
    """Load model + scaler + feature-list for one target."""
    stem   = MODEL_DIR / f"{ticker}_{target}"

    ckpt      = stem.with_suffix(".pt")
    scaler    = joblib.load(stem.with_name(f"{stem.name}_scaler.pkl"))
    feats     = joblib.load(stem.with_name(f"{stem.name}_features.pkl"))
    hp_path   = stem.with_suffix(".json")

    hp = json.loads(hp_path.read_text())         # hyper-params from Optuna / training
    model_type = hp.pop("model_type", "TransformerTCN")  # נשלוף ונוציא כדי שלא יעבור פעמיים

    # --- בונים את הרשת ---
    net = get_model(
        len(feats),          # input_dim (n_features)
        model_type,          # TransformerTCN / LSTM / …
        1,                   # output_dim (positional!)
        **hp                 # שאר ההייפר-פרמטרים
    ).cpu()

    net.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    net.eval()

    return net, scaler, feats