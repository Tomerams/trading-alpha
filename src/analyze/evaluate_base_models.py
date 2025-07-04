#!/usr/bin/env python3
"""
Evaluate every base-model *against its own target column*.
מדפיס MAE / RMSE / R² לכל טארגט +  Dir-Acc (סימן) לטווחי 1-5 ימים.
"""
from datetime import date, timedelta
from pathlib import Path
import json, joblib, numpy as np, pandas as pd, torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    brier_score_loss,
    mean_absolute_percentage_error,
)

from config.meta_data_config import META_PARAMS
from data.data_processing import get_indicators_data
from models.model_utilities import get_model
from routers.routers_entities import UpdateIndicatorsData


# ── קבועים ────────────────────────────────────────────────────────────
TICKER = "QQQ"
BASE_TARGETS = META_PARAMS["base_targets"]  # 11 targets
SEQ_LEN = META_PARAMS.get("seq_len", 60)
MODEL_DIR = Path("src/files/models")


def classification_metrics(y_true, y_pred, thr=0.0):
    """
    מחזיר dict עם precision/recall/F1 + brier + confusion-matrix
    עבור תחזית סימן (up / down).
    """
    y_bin = (y_true > 0).astype(int)  # 1 = Up, 0 = Down/Flat
    yhat_bin = (y_pred > thr).astype(int)

    return {
        "precision": precision_score(y_bin, yhat_bin, zero_division=0),
        "recall": recall_score(y_bin, yhat_bin, zero_division=0),
        "f1": f1_score(y_bin, yhat_bin, zero_division=0),
        "brier": brier_score_loss(y_bin, yhat_bin),
        "cm": confusion_matrix(y_bin, yhat_bin).tolist(),  # לשמירה כ-JSON
    }


def get_data():
    """Get and prepare the data"""
    req = UpdateIndicatorsData(
        stock_ticker=TICKER,
        start_date=(date.today() - timedelta(days=365 * 20)).isoformat(),
        end_date=date.today().isoformat(),
        indicators=[],
        scale=False,
    )
    return get_indicators_data(req).dropna().reset_index(drop=True)


def _guess_arch_params(chkpt: dict) -> dict:
    """Fallback – שולף hidden_size / num_layers מה-state-dict כאשר
    קובץ-הייפרים *.json* לא נמצא."""
    try:
        hidden = chkpt["net.tcn.network.0.conv1.bias"].shape[0]
        # כל בלוק TCN מוסיף שני קונבולוציות; נחפש כמה Layers מופיעים
        n_layers = (
            max(int(k.split(".")[3]) for k in chkpt if k.startswith("net.tcn.network.")) + 1
        )
        return {"hidden_size": hidden, "num_layers": n_layers}
    except (KeyError, IndexError, ValueError) as e:
        print(f"Warning: Could not guess architecture parameters: {e}")
        return {"hidden_size": 128, "num_layers": 4}  # Default values


def _load_predict(target: str, df: pd.DataFrame) -> np.ndarray:
    """
    בונה את הטנסור בצורת (batch, F, 60) בדיוק עם מספר-הפיצ'רים
    שה-TCN אומן עליו, מריץ את המודל ומחזיר חיזוי וקטורי.
    """
    stem = MODEL_DIR / f"{TICKER}_{target}"
    
    # Check if model files exist
    model_file = stem.with_suffix(".pt")
    features_file = stem.parent / f"{stem.name}_features.pkl"
    scaler_file = stem.parent / f"{stem.name}_scaler.pkl"
    
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    if not features_file.exists():
        raise FileNotFoundError(f"Features file not found: {features_file}")
    if not scaler_file.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_file}")
    
    # Load model checkpoint
    ckpt = torch.load(model_file, map_location="cpu")

    # Load the full feature list the model was trained with
    feats = joblib.load(features_file)
    scaler = joblib.load(scaler_file)
    
    # Check if target column exists in dataframe
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe")
    
    # Check if all required features exist in dataframe
    missing_features = [f for f in feats if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in dataframe: {missing_features}")
    
    # Get output dimension
    try:
        out_dim = ckpt["net.head.weight"].shape[0]
    except KeyError:
        print("Warning: Could not determine output dimension from checkpoint, using 1")
        out_dim = 1

    # Load hyperparameters
    hp_file = stem.with_suffix(".json")
    if hp_file.exists():
        try:
            hp = json.loads(hp_file.read_text())
            hp.pop("lr", None)  # Remove learning rate if present
            hp["hidden_size"] = hp.pop("hidden", hp.get("hidden_size", 128))
            hp["num_layers"] = hp.pop("n_layers", hp.get("num_layers", 4))
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Error loading hyperparameters from {hp_file}: {e}")
            hp = _guess_arch_params(ckpt)
    else:
        hp = _guess_arch_params(ckpt)

    # Initialize the model with the correct number of features
    try:
        model = get_model(len(feats), "TransformerTCN", out_dim, **hp)
        model.load_state_dict(ckpt, strict=False)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Error initializing model for target {target}: {e}")

    # Build tensor (n, SEQ_LEN, F) - prepare for potential TCN internal transpose
    try:
        # Check if we have enough data
        if len(df) < SEQ_LEN + 1:
            raise ValueError(f"Insufficient data: need at least {SEQ_LEN + 1} rows, got {len(df)}")
            
        # Debug info
        print(f"Target: {target}")
        print(f"Expected features: {len(feats)}")
        print(f"Available features in df: {len([f for f in feats if f in df.columns])}")
        
        # Ensure we only use the features that the model was trained on
        if not all(f in df.columns for f in feats):
            missing = [f for f in feats if f not in df.columns]
            raise ValueError(f"Missing required features: {missing}")
            
        data_subset = df[feats].values  # Shape: (n_rows, n_features)
        print(f"Data subset shape: {data_subset.shape}")
            
        # Manual sliding window creation
        n_samples = len(data_subset) - SEQ_LEN + 1
        X = np.zeros((n_samples - 1, SEQ_LEN, len(feats)))  # -1 to exclude last sample
            
        for i in range(n_samples - 1):
            X[i] = data_subset[i:i + SEQ_LEN]
            
        print(f"Sliding window shape: {X.shape}") # This will be (batch, SEQ_LEN, features)

        # Reshape for scaling: (n_samples * seq_len, n_features)
        X_reshaped = X.reshape(-1, len(feats))
        X_scaled = scaler.transform(X_reshaped)
            
        # Reshape back: (n_samples, seq_len, n_features)
        X = X_scaled.reshape(-1, SEQ_LEN, len(feats))
            
        # *** REMOVED THE TRANSPOSE OPERATION HERE ***
        # X = X.transpose(0, 2, 1).astype(np.float32)  # (n_samples, n_features, seq_len)
        X = X.astype(np.float32) # Keep as (n_samples, SEQ_LEN, n_features)
            
        print(f"Final input shape before model: {X.shape}")
        print(f"Assuming model expects (batch_size, sequence_length, channels)")
            
    except Exception as e:
        raise RuntimeError(f"Error preparing input data for target {target}: {e}")

    # Predict
    try:
        with torch.no_grad():
            pred = model(torch.tensor(X)).numpy()
            
        return pred.squeeze() if out_dim == 1 else pred[:, 0]
    except Exception as e:
        raise RuntimeError(f"Error during prediction for target {target}: {e}")


def _rmse(y, yhat):
    """Calculate RMSE with backward compatibility"""
    try:
        return mean_squared_error(y, yhat, squared=False)
    except TypeError:  # sklearn<0.22
        return np.sqrt(mean_squared_error(y, yhat))


def dir_acc(y, yhat, thr=0.0):
    """Calculate directional accuracy"""
    return accuracy_score(np.sign(y), np.sign(yhat - thr))


def main():
    """Main evaluation function"""
    print("Loading data...")
    try:
        df_raw = get_data()
        print(f"Data loaded: {len(df_raw)} rows, {len(df_raw.columns)} columns")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    print(f"\nEvaluating {len(BASE_TARGETS)} targets...")
    print("\nTarget evaluation")
    print("-" * 80)
        
    successful_evaluations = 0
    failed_evaluations = []
        
    for tgt in BASE_TARGETS:
        try:
            # Get true values
            if tgt not in df_raw.columns:
                print(f"Skipping {tgt}: target column not found in data")
                failed_evaluations.append(tgt)
                continue
                
            y_true = df_raw[tgt].iloc[SEQ_LEN:].values
                
            # Get predictions
            y_pred = _load_predict(tgt, df_raw)
                
            # Check if predictions match expected length
            if len(y_pred) != len(y_true):
                print(f"Warning: Length mismatch for {tgt}: pred={len(y_pred)}, true={len(y_true)}")
                min_len = min(len(y_pred), len(y_true))
                y_pred = y_pred[:min_len]
                y_true = y_true[:min_len]
                
            # Calculate metrics
            mae = mean_absolute_error(y_true, y_pred)
            rmse = _rmse(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            mape = mean_absolute_percentage_error(y_true, y_pred)
                
            cls = classification_metrics(y_true, y_pred, thr=0.0)
                
            print(
                f"{tgt:<22}  MAE={mae:.4f}  RMSE={rmse:.4f}  "
                f"R²={r2:.3f}  MAPE={mape:.2%}  "
                f"P={cls['precision']:.3f}  R={cls['recall']:.3f}  "
                f"F1={cls['f1']:.3f}  Brier={cls['brier']:.3f}"
            )
            successful_evaluations += 1
                
        except Exception as e:
            print(f"Error evaluating {tgt}: {e}")
            failed_evaluations.append(tgt)
        
    # Directional accuracy for first 5 targets
    print(f"\nDirectional-Accuracy (sign) - First 5 targets")
    print("-" * 60)
        
    for tgt in BASE_TARGETS[:5]:  # 1-5 day horizons
        try:
            if tgt in failed_evaluations:
                print(f"{tgt:<22}  FAILED")
                continue
                    
            y_true = df_raw[tgt].iloc[SEQ_LEN:].values
            y_pred = _load_predict(tgt, df_raw)
                    
            # Handle length mismatch
            if len(y_pred) != len(y_true):
                min_len = min(len(y_pred), len(y_true))
                y_pred = y_pred[:min_len]
                y_true = y_true[:min_len]
                    
            da = dir_acc(y_true, y_pred)
            print(f"{tgt:<22}  {da:.3f}")
                    
        except Exception as e:
            print(f"{tgt:<22}  ERROR: {e}")
        
    print(f"\nSummary: {successful_evaluations}/{len(BASE_TARGETS)} targets evaluated successfully")
    if failed_evaluations:
        print(f"Failed targets: {failed_evaluations}")


if __name__ == "__main__":
    main()