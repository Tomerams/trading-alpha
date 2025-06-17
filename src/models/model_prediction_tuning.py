#  PYTHONPATH=./src python3 src/models/model_prediction_tuning.py

"""
Optuna tuning *פר-טארגט*  – v1 (June 2025)
⟶ מריץ Optuna  לכל Target בסיסי,  שומר best-params כ-JSON נפרד.
   תוצאה: 11 קבצי  files/models/best_params_<Target>.json
   אותם ניתן לטעון לפני train_single / train_all_base_models.

▪ Directional-Accuracy (thr 5 %) היא מטריקת האופטימיזציה.
▪ משותפת הכנת DataFrame / Scaler, כדי לחסוך זמן.
▪ ניתן לשנות n_trials / timeout ב-OPTUNA_PARAMS.
"""
from __future__ import annotations
import json, time, joblib, optuna, numpy as np, torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from config.model_trainer_config import MODEL_TRAINER_PARAMS, TRAIN_TARGETS_PARAMS
from config.optimizations_config import OPTUNA_PARAMS
from data.data_processing import get_indicators_data
from models.model_utilities import get_model, time_based_split, create_sequences
from routers.routers_entities import UpdateIndicatorsData

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────────────────– DirAcc helper ──────────────────────────────

def dir_acc(y_true: np.ndarray, y_pred: np.ndarray, thr: float = 0.05) -> float:
    signal = np.where(y_pred > thr, 1, np.where(y_pred < -thr, -1, 0))
    return accuracy_score(np.sign(y_true), signal)

# ───────────────────── Prepare shared tensors once ───────────────────

def prepare_shared(req, seq_len):
    df = get_indicators_data(req).dropna().reset_index(drop=True)
    tr_df, va_df, _ = time_based_split(df)
    feats = [c for c in df.columns if c not in {"Date", *TRAIN_TARGETS_PARAMS["target_cols"]}]

    # full feature windows – same for all targets
    X_tr, _ = create_sequences(tr_df, feats, [TRAIN_TARGETS_PARAMS["target_cols"][0]], seq_len)
    X_va, _ = create_sequences(va_df, feats, [TRAIN_TARGETS_PARAMS["target_cols"][0]], seq_len)

    scaler = StandardScaler().fit(X_tr.reshape(-1, X_tr.shape[-1]))
    X_tr = scaler.transform(X_tr.reshape(-1, X_tr.shape[-1])).reshape(X_tr.shape)
    X_va = scaler.transform(X_va.reshape(-1, X_va.shape[-1])).reshape(X_va.shape)

    return X_tr, X_va, feats

REQ = UpdateIndicatorsData(ticker="QQQ", start_date="2005-01-01", end_date="2025-06-17", indicators=[])
SEQ_LEN = MODEL_TRAINER_PARAMS["seq_len"]
X_TR_SHARED, X_VA_SHARED, FEATURE_COLS = prepare_shared(REQ, SEQ_LEN)
FEAT_DIM = X_TR_SHARED.shape[-1]

# create directory for outputs
Path("files/models").mkdir(parents=True, exist_ok=True)

# ───────────────────────── Objective per target ──────────────────────

def build_objective(y_tr, y_va):
    tr_ds = TensorDataset(torch.tensor(X_TR_SHARED), torch.tensor(y_tr))
    va_ds = TensorDataset(torch.tensor(X_VA_SHARED), torch.tensor(y_va))
    tr_loader = DataLoader(tr_ds, batch_size=OPTUNA_PARAMS["batch_size"], shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=OPTUNA_PARAMS["batch_size"]*2)

    def objective(trial: optuna.Trial) -> float:
        hid = trial.suggest_categorical("hidden", [64, 96, 128, 192, 256])
        nl  = trial.suggest_int("n_layers", 2, 4)
        dr  = trial.suggest_float("dropout", 0.0, 0.3, step=0.05)
        lr  = trial.suggest_float("lr", 5e-4, 3e-3, log=True)

        model = get_model(FEAT_DIM, MODEL_TRAINER_PARAMS["model_type"], 1,  # output_size=1
                          hidden_size=hid, num_layers=nl, dropout=dr).to(DEVICE)
        opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=OPTUNA_PARAMS.get("max_epochs", 25))
        loss_fn = torch.nn.MSELoss()

        best_acc, patience = 0.0, 0
        for ep in range(1, OPTUNA_PARAMS["max_epochs"]+1):
            model.train()
            for xb, yb in tr_loader:
                xb, yb = xb.to(DEVICE).float(), yb.to(DEVICE).float()
                opt.zero_grad(set_to_none=True)
                loss_fn(model(xb), yb).backward(); opt.step()
            sched.step()

            # validation DirAcc
            model.eval(); preds = []
            with torch.no_grad():
                for xb, _ in va_loader:
                    preds.append(model(xb.to(DEVICE).float()).cpu())
            acc = dir_acc(y_va.squeeze(), torch.cat(preds).squeeze().numpy())

            trial.report(acc, ep)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if acc > best_acc + 1e-4: best_acc, patience = acc, 0
            else:
                patience += 1
                if patience >= OPTUNA_PARAMS["early_stopping_patience"]: break

        return 1.0 - best_acc   # minimize

    return objective

# ───────────────────────── Loop over targets ─────────────────────────

def tune_all_targets():
    results = {}
    for tgt in TRAIN_TARGETS_PARAMS["target_cols"]:
        print(f"\n── Optuna tuning for {tgt} ─────────────────────────────────")
        # build target arrays
        df = get_indicators_data(REQ).dropna().reset_index(drop=True)
        tr_df, va_df, _ = time_based_split(df)
        _, y_tr = create_sequences(tr_df, FEATURE_COLS, [tgt], SEQ_LEN)
        _, y_va = create_sequences(va_df, FEATURE_COLS, [tgt], SEQ_LEN)

        study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=3))
        study.optimize(build_objective(y_tr.astype(np.float32), y_va.astype(np.float32)),
                       n_trials=OPTUNA_PARAMS["n_trials"],
                       timeout=OPTUNA_PARAMS.get("timeout_seconds"))

        best_params = study.best_params
        best_acc    = 1.0 - study.best_value
        print(f"🏆  {tgt}: Best DirAcc = {best_acc:.3f}  |  params = {best_params}")

        # persist
        fname = Path("files/models") / f"best_params_{tgt}.json"
        fname.write_text(json.dumps(best_params, indent=2))
        joblib.dump(study, fname.with_suffix(".pkl"))
        results[tgt] = {"dir_acc": best_acc, **best_params}
    return results

if __name__ == "__main__":
    summary = tune_all_targets()
    Path("files/models/optuna_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n🔸 All targets tuned – summary saved to optuna_summary.json")