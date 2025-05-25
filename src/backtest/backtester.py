import logging
import time
import math
from typing import Any, Dict, List, Tuple
import os
import joblib
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import numpy as np
import pandas as pd
import torch

from config.backtest_config import BACKTEST_PARAMS #
from config.model_trainer_config import MODEL_TRAINER_PARAMS #
from config.meta_data_config import META_PARAMS #
from data.data_processing import get_indicators_data #
from models.model_utilities import load_model #

# Configure root logger
def _get_logger() -> logging.Logger:
    # Changed logger name to be specific to this module for clarity
    log = logging.getLogger("backtest.backtester") 
    if not log.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s" # Added logger name
        )
        handler.setFormatter(formatter)
        log.addHandler(handler)
        log.setLevel(logging.INFO)
    return log

logger = _get_logger()


def _load_deep_base(ticker: str) -> Tuple[Any, Any, List[str], int]:
    """Load TransformerTCN model, scaler, feature columns, and sequence length."""
    logger.info("⏳ [Backtest] Loading deep base-model for %s…", ticker)
    start_time_load_deep = time.time()
    
    model, scaler, feature_cols = load_model( #
        ticker, MODEL_TRAINER_PARAMS['model_type'] #
    )
    seq_len = MODEL_TRAINER_PARAMS['seq_len'] #
    
    elapsed_time = time.time() - start_time_load_deep
    logger.info(
        "✅ [Backtest] Deep base model loaded in %.2f s. Features count: %d, Seq length: %d",
        elapsed_time,
        len(feature_cols),
        seq_len
    )
    return model, scaler, feature_cols, seq_len


def _get_meta_pipeline(timeout: float = 10.0) -> Tuple[Dict[str, Any], Any]:
    """Load meta-model pipeline with a timeout to avoid hangs, using joblib."""
    path = META_PARAMS.get('meta_model_path') #
    if not path:
        logger.error("❌ [Backtest] Meta model path not found in META_PARAMS. Cannot load meta pipeline.")
        return {}, None # החזרת ערכים ריקים אם אין נתיב
        
    logger.info("⏳ [Backtest] Attempting to load meta pipeline with timeout (%.1fs) from %s using joblib…", timeout, path)

    def _load_with_joblib(): # שם פנימי עודכן
        if not os.path.exists(path):
            logger.error(f"❌ [Backtest - _load_with_joblib] Meta pipeline file NOT FOUND: {path}")
            raise FileNotFoundError(f"Meta pipeline file not found during _load_with_joblib: {path}")
        
        logger.info(f"⏳ [Backtest - _load_with_joblib] Attempting joblib.load from: {path}")
        pipeline = joblib.load(path) # *** שימוש ב-joblib.load ***
        logger.info(f"✅ [Backtest - _load_with_joblib] Meta pipeline successfully loaded with joblib from: {path}")
        
        base_models_loaded = pipeline.get('base', {})
        meta_classifier_loaded = pipeline.get('meta')
        
        if not meta_classifier_loaded: # בדיקה אם המפתח 'meta' קיים והערך אינו None
            logger.warning("⚠️ [Backtest - _load_with_joblib] Meta classifier ('meta' key) not found or is None in the loaded pipeline.")
        if not base_models_loaded: # בדיקה אם המפתח 'base' קיים והערך אינו ריק
            logger.warning("⚠️ [Backtest - _load_with_joblib] Base models ('base' key) not found or is empty in the loaded pipeline.")
            
        return base_models_loaded, meta_classifier_loaded

    load_start_time = time.time()
    base_models_from_pipeline = {} # אתחול כברירת מחדל
    meta_classifier_from_pipeline = None # אתחול כברירת מחדל

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_load_with_joblib)
        try:
            base_models_from_pipeline, meta_classifier_from_pipeline = future.result(timeout=timeout)
            elapsed_load_time = time.time() - load_start_time
            
            # גם אם אחד מהם ריק, נרצה לרשום זאת ולא בהכרח כשגיאה קריטית כאן
            if not meta_classifier_from_pipeline or not base_models_from_pipeline:
                logger.warning( # שונה ל-warning כי ייתכן שהקוד בהמשך יודע להתמודד עם זה
                    "⚠️ [Backtest] Meta pipeline loaded with joblib but seems empty or incomplete (loaded in %.2f s): base models count=%d, meta_classifier type=%s",
                    elapsed_load_time,
                    len(base_models_from_pipeline), # יהיה 0 אם ריק
                    type(meta_classifier_from_pipeline).__name__ if meta_classifier_from_pipeline else "None",
                )
            else:
                logger.info(
                    "✅ [Backtest] Meta pipeline loaded successfully with joblib in %.2f s (%d base models, meta_classifier type: %s)",
                    elapsed_load_time,
                    len(base_models_from_pipeline),
                    type(meta_classifier_from_pipeline).__name__,
                )
        except TimeoutError:
            elapsed_timeout_time = time.time() - load_start_time
            logger.error(
                "❌ [Backtest] Meta pipeline load (joblib) TIMED OUT after %.2f seconds (configured timeout: %.1fs). Returning empty meta model.",
                elapsed_timeout_time,
                timeout,
            )
            base_models_from_pipeline, meta_classifier_from_pipeline = {}, None # החזרת ערכים ריקים במקרה של timeout
        except FileNotFoundError as fnf_error: # תפיסת שגיאת קובץ לא נמצא מהפונקציה הפנימית
            logger.error(f"❌ [Backtest] FileNotFoundError during meta pipeline load: {fnf_error}. Returning empty meta model.", exc_info=True)
            base_models_from_pipeline, meta_classifier_from_pipeline = {}, None
        except Exception as e:
            elapsed_error_time = time.time() - load_start_time
            logger.error(f"❌ [Backtest] Meta pipeline load (joblib) encountered an unexpected error after {elapsed_error_time:.2f}s. Returning empty meta model.", exc_info=True)
            base_models_from_pipeline, meta_classifier_from_pipeline = {}, None
    
    return base_models_from_pipeline, meta_classifier_from_pipeline


def _prepare_data(
    request_data: Any,
    feature_cols: List[str],
    scaler: Any,
    seq_len: int
) -> Tuple[pd.Series, pd.Series, np.ndarray]:
    """Fetch indicator data, build sequences, and scale features."""
    logger.info("⏳ [Backtest] Fetching indicator data for backtest data preparation…")
    start_fetch_time = time.time()
    df = get_indicators_data(request_data) #
    if df.empty:
        logger.error("❌ [Backtest - _prepare_data] Indicator data is empty after fetching. Cannot proceed.")
        # Consider how to handle this - raising an error might be best.
        raise ValueError("Indicator data for backtest preparation is empty.")
    logger.info("✅ [Backtest - _prepare_data] Raw indicator data fetched in %.2f s (original rows=%d)", time.time() - start_fetch_time, len(df))

    df = df.dropna().reset_index(drop=True)
    if df.empty:
        logger.error("❌ [Backtest - _prepare_data] Dataframe is empty after dropna(). Cannot proceed.")
        raise ValueError("Dataframe became empty after dropna() in _prepare_data.")

    if 'Date' not in df.columns:
        logger.error("❌ [Backtest - _prepare_data] 'Date' column missing after dropna and reset_index.")
        raise KeyError("'Date' column missing in prepared dataframe for backtest.")
    df['Date'] = pd.to_datetime(df['Date'])
    logger.info("✅ [Backtest - _prepare_data] Data cleaned (dropna, reset_index, Date to_datetime). Current shape: %s", df.shape)

    if len(df) < seq_len : 
        logger.error(f"❌ [Backtest - _prepare_data] Cleaned data length ({len(df)}) is less than seq_len ({seq_len}). Cannot create sequences.")
        raise ValueError(f"Cleaned data length {len(df)} is less than seq_len {seq_len} for sequence creation.")

    # Sequence creation
    start_seq_time = time.time()
    num_raw_sequences = len(df) - seq_len # This many sequences will be used for PREDICTIONS
                                        # The corresponding targets start at index 'seq_len'

    if num_raw_sequences < 0: # Should be caught by len(df) < seq_len, but defensive
        num_raw_sequences = 0
    
    if num_raw_sequences > 0 :
        logger.info("⏳ [Backtest - _prepare_data] Creating %d sequences for predictions...", num_raw_sequences)
        X = np.stack([
            df[feature_cols].iloc[i : i + seq_len].values
            for i in range(num_raw_sequences) 
        ])
        # Align dates and prices with where the *targets* for these X sequences would be
        # If X[k] uses df rows k to k+seq_len-1, its target is at df row k+seq_len
        # So, dates and prices should start from index seq_len and have num_raw_sequences items
        idx_start = seq_len
        idx_end = seq_len + num_raw_sequences
        
        dates = df['Date'].iloc[idx_start:idx_end].reset_index(drop=True)
        prices = df['Close'].iloc[idx_start:idx_end].reset_index(drop=True)
    else:
        logger.warning("⚠️ [Backtest - _prepare_data] No sequences to create (num_raw_sequences is %d). X will be empty.", num_raw_sequences)
        # Define X with correct number of features for scaler if it's empty
        num_features = len(feature_cols) if feature_cols else (scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else 1)
        X = np.array([]).reshape(0, seq_len if seq_len > 0 else 1, num_features) # Ensure 3D
        dates = pd.Series([], dtype='datetime64[ns]')
        prices = pd.Series([], dtype='float64')

    logger.info("✅ [Backtest - _prepare_data] Sequences created in %.2f s. X shape: %s, Dates len: %d, Prices len: %d", 
                time.time() - start_seq_time, X.shape, len(dates), len(prices))
    
    # Scaling
    start_scale_time = time.time()
    X_scaled = np.array([]) # Default to empty
    if X.size > 0 :
        original_shape = X.shape
        if X.ndim == 3: # Expected: (num_sequences, seq_len, num_features)
            num_features_in_x = X.shape[2]
            X_reshaped_for_scaler = X.reshape(-1, num_features_in_x)
            X_scaled_flat = scaler.transform(X_reshaped_for_scaler)
            X_scaled = X_scaled_flat.reshape(original_shape)
        elif X.ndim == 2 and seq_len <=1 : # Case for seq_len=0 or 1 where X might be 2D (samples, features)
             logger.warning("⚠️ [Backtest - _prepare_data] X is 2D for scaling, assuming seq_len was effectively 1.")
             X_scaled_transformed = scaler.transform(X)
             # We need to ensure X_scaled is 3D if the deep model expects it
             if MODEL_TRAINER_PARAMS.get('seq_len', 1) > 0: # Check config
                 X_scaled = X_scaled_transformed[:, np.newaxis, :] # Add seq_len dimension of 1
             else:
                 X_scaled = X_scaled_transformed # Or keep as 2D if model handles it
        else:
            logger.error(f"❌ [Backtest - _prepare_data] X has unexpected ndim: {X.ndim} (shape {X.shape}) for scaling with seq_len {seq_len}.")
            X_scaled = X # Pass through if shape is wrong, or raise error
        logger.info("✅ [Backtest - _prepare_data] Data scaled in %.2f s. X_scaled shape: %s", time.time() - start_scale_time, X_scaled.shape)
    else:
        # X was empty, X_scaled remains empty (with correct feature dimension if possible)
        num_features = len(feature_cols) if feature_cols else (scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else 1)
        X_scaled = np.array([]).reshape(0, seq_len if seq_len > 0 else 1, num_features)
        logger.info("ℹ️ [Backtest - _prepare_data] X was empty, X_scaled is also empty. Shape: %s", X_scaled.shape)

    # Final alignment check
    if X_scaled.shape[0] != len(dates) or X_scaled.shape[0] != len(prices):
        logger.warning(f"⚠️ [Backtest - _prepare_data] Mismatch after scaling: X_scaled ({X_scaled.shape[0]}), dates ({len(dates)}), prices ({len(prices)}). Attempting to truncate.")
        min_len = min(X_scaled.shape[0], len(dates), len(prices))
        X_scaled = X_scaled[:min_len]
        dates = dates[:min_len]
        prices = prices[:min_len]
        logger.info(f"ℹ️ [Backtest - _prepare_data] All data components (X_scaled, dates, prices) adjusted to min_len: {min_len}")

    return dates, prices, X_scaled


def _infer_deep(model: Any, X_scaled: np.ndarray) -> np.ndarray:
    """Run deep model inference and return predictions."""
    if X_scaled.size == 0:
        logger.warning("⚠️ [Backtest - _infer_deep] X_scaled is empty. Returning empty array for deep predictions.")
        num_model_outputs = len(TRAIN_TARGETS_PARAMS.get("target_cols", [1])) #
        return np.array([]).reshape(0, num_model_outputs)

    logger.info("⏳ [Backtest - _infer_deep] Running deep-model inference on %d samples (X_scaled shape: %s)…", X_scaled.shape[0], X_scaled.shape)
    start_infer_time = time.time()
    
    tensor = torch.tensor(X_scaled, dtype=torch.float32)
    # The _prepare_data function should ensure X_scaled is 3D (batch, seq, features) if seq_len > 0.
    # If seq_len was 0 or 1 and X_scaled is (batch, features), some models might need unsqueeze(1) for seq_len dim.
    # Original code: if tensor.ndim == 2: tensor = tensor.unsqueeze(1)
    # This is generally safe if the model expects a sequence dimension, even if it's 1.
    # Let's assume models are flexible or expect (batch, seq_len, features)
    if tensor.ndim == 2 and MODEL_TRAINER_PARAMS.get('model_type') not in ['SomeNonSequentialModel']: # Example condition
        # This depends on whether your model (e.g. TransformerTCN) can handle 2D input if seq_len is 1.
        # Typically, sequence models expect 3D input.
        # If X_scaled is (N, F) and it should be (N, 1, F):
        # tensor = tensor.unsqueeze(1)
        # For now, let's assume _prepare_data provides the correct shape.
        pass


    with torch.no_grad():
        preds = model(tensor).cpu().numpy()
    logger.info("✅ [Backtest - _infer_deep] Deep inference done in %.2f s. Predictions shape: %s", time.time() - start_infer_time, preds.shape)
    return preds


def _simulate_trades(
    dates: pd.Series,
    prices: pd.Series,
    deep_preds: np.ndarray,
    meta_base: Dict[str, Any], # Renamed for clarity
    meta_clf: Any # Renamed for clarity
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Simulate trades using meta-model (if available) and stop rules."""

    # Ensure all inputs for simulation have the same length for iteration
    # deep_preds rows should match len(dates) and len(prices) for decisions
    # The simulation loop goes up to len(dates)-1, so deep_preds should have at least that many.
    num_decision_points = len(dates)
    if len(prices) != num_decision_points or deep_preds.shape[0] != num_decision_points :
        logger.warning(f"⚠️ [Backtest Sim] Length mismatch: dates ({len(dates)}), prices ({len(prices)}), deep_preds ({deep_preds.shape[0]}). Truncating to shortest for simulation.")
        min_len = min(len(dates), len(prices), deep_preds.shape[0])
        if min_len <=1 : # Need at least one point to make a decision for the next, and a final point for sell-off
             logger.error("❌ [Backtest Sim] Not enough aligned data points (<=1) after attempting to resolve mismatch. Cannot simulate.")
             return [], {'ticker_change': 0.0, 'net_profit': 0.0, 'num_trades': 0, 'error': "Insufficient aligned data"}
        dates = dates[:min_len]
        prices = prices[:min_len]
        deep_preds = deep_preds[:min_len]
        num_decision_points = min_len
        logger.info(f"ℹ️ [Backtest Sim] Adjusted to {num_decision_points} decision points.")


    if num_decision_points <= 1: # If only one data point (or none), cannot simulate trades effectively
        logger.warning("⚠️ [Backtest Sim] Not enough data points (%d) to simulate trades. Returning empty results.", num_decision_points)
        return [], {'ticker_change': 0.0, 'net_profit': 0.0, 'num_trades': 0}

    logger.info("⏳ [Backtest Sim] Starting trade simulation for %d decision points (up to %d trades possible)…", num_decision_points, num_decision_points -1)
    start_sim_time = time.time()

    cash = BACKTEST_PARAMS['initial_balance'] #
    shares = 0
    last_buy_price = 0.0 # Price of last BUY transaction
    current_peak_price_since_buy = 0.0 # For trailing stop on price

    trades_log: List[Dict[str, Any]] = [] # Changed variable name from 'trades'
    
    # We make decisions at step 'i' using data from 'i', to act at price 'i' (or effectively, open of i+1)
    # The loop should go up to one before the last data point, as the last point is only for potential final sell-off.
    for i in range(num_decision_points - 1): 
        current_date = dates.iloc[i]
        current_price = float(prices.iloc[i])

        if i > 0 and i % 100 == 0: # Log progress every 100 steps
            logger.info("  [Backtest Sim] Processed %d/%d bars… Current Portfolio Value: %.2f", i, num_decision_points - 1, cash + shares * current_price)
        
        action_to_take = 1 # Default to HOLD (1)
        if meta_clf and meta_base: # Check if meta_base is not empty
            current_deep_pred_features = deep_preds[i].reshape(1, -1) 
            try:
                base_model_probabilities = []
                for model_name, base_model_instance in meta_base.items():
                    proba_vector = base_model_instance.predict_proba(current_deep_pred_features)
                    base_model_probabilities.append(proba_vector.flatten())
                
                final_meta_features = np.concatenate(base_model_probabilities).reshape(1, -1)
                action_to_take = int(meta_clf.predict(final_meta_features)[0])
            except Exception as e:
                logger.error(f"  [Backtest Sim] Error during meta-model prediction at step {i} for date {current_date.isoformat()}: {e}. Defaulting to HOLD.", exc_info=False) # exc_info=False for less verbose default
                action_to_take = 1 
        elif deep_preds.ndim > 1 and deep_preds.shape[0] > i and deep_preds.shape[1] > 0 :
            action_to_take = int(np.argmax(deep_preds[i]))
        else:
            logger.warning(f"  [Backtest Sim] No meta model and deep_preds not usable at step {i} (shape: {deep_preds.shape}). Defaulting to HOLD.")
            action_to_take = 1

        # Trading Logic
        if shares > 0: 
            current_peak_price_since_buy = max(current_peak_price_since_buy, current_price)
            sell_triggered = False
            sell_reason = ""
            
            if action_to_take == 0: 
                sell_triggered = True
                sell_reason = "ACTION_SIGNAL_SELL"
            elif current_price <= last_buy_price * (1 - BACKTEST_PARAMS['stop_loss_pct']): #
                sell_triggered = True
                sell_reason = "STOP_LOSS"
            elif current_peak_price_since_buy > 0 and \
                 current_price <= current_peak_price_since_buy * (1 - BACKTEST_PARAMS['trailing_stop_pct']): #
                sell_triggered = True
                sell_reason = "TRAILING_STOP"
            
            if sell_triggered:
                fee = max(BACKTEST_PARAMS['minimum_fee'], shares * BACKTEST_PARAMS['buy_sell_fee_per_share']) #
                profit_before_tax = (current_price - last_buy_price) * shares
                tax = BACKTEST_PARAMS['tax_rate'] * max(0, profit_before_tax) #
                cash += shares * current_price - fee - tax
                
                trades_log.append({
                    'Date': current_date.isoformat(), 'Type': 'SELL', 'Price': current_price, 
                    'Shares': shares, 'Cash': cash, 'Portfolio': cash, 
                    'Reason': sell_reason, 'Fee': fee, 'Tax': tax, 'Profit_Before_Tax': profit_before_tax
                })
                logger.info(f"  [Backtest Sim] SELL {shares} @ {current_price:.2f} on {current_date.date()} ({sell_reason}). Cash: {cash:.2f}")
                shares = 0
                current_peak_price_since_buy = 0 
        
        elif action_to_take == 2 and shares == 0: 
            buy_fee_per_share = BACKTEST_PARAMS.get('buy_sell_fee_per_share_buy', BACKTEST_PARAMS['buy_sell_fee_per_share']) #
            min_buy_fee = BACKTEST_PARAMS.get('minimum_fee_buy', BACKTEST_PARAMS['minimum_fee']) #
            
            # Calculate actual fee based on potential shares later
            # For now, estimate max possible shares to see if any can be bought
            if current_price <= 0: # Avoid division by zero
                logger.warning(f"  [Backtest Sim] Price is {current_price} on {current_date.date()}, cannot BUY.")
                continue

            # Tentatively calculate shares without exact fee, then refine
            potential_shares = math.floor(cash / current_price) # Max shares if no fee
            if potential_shares > 0:
                estimated_fee = max(min_buy_fee, potential_shares * buy_fee_per_share)
                if cash > estimated_fee: # Check if cash can cover at least the fee
                    shares_can_buy = math.floor((cash - min_buy_fee) / (current_price + buy_fee_per_share)) # More precise share calculation
                    # Or, simpler: shares_can_buy = math.floor((cash - min_buy_fee) / current_price) and then calculate actual fee
                    
                    if shares_can_buy > 0:
                        actual_fee = max(min_buy_fee, shares_can_buy * buy_fee_per_share)
                        total_cost = shares_can_buy * current_price + actual_fee

                        if cash >= total_cost:
                            cash -= total_cost
                            shares = shares_can_buy
                            last_buy_price = current_price 
                            current_peak_price_since_buy = current_price

                            trades_log.append({
                                'Date': current_date.isoformat(), 'Type': 'BUY', 'Price': current_price, 
                                'Shares': shares, 'Cash': cash, 'Portfolio': cash + shares * current_price,
                                'Fee': actual_fee
                            })
                            logger.info(f"  [Backtest Sim] BUY {shares} @ {current_price:.2f} on {current_date.date()}. Cash: {cash:.2f}")
                        else:
                             logger.info(f"  [Backtest Sim] Signal BUY at {current_price:.2f}, but not enough cash ({cash:.2f}) for {shares_can_buy} shares + fee ({actual_fee:.2f}).")
                    else:
                        logger.info(f"  [Backtest Sim] Signal BUY at {current_price:.2f}, but not enough cash to buy any shares after considering fees.")
                else:
                    logger.info(f"  [Backtest Sim] Signal BUY at {current_price:.2f}, but not enough cash for minimum fee.")
            else:
                 logger.info(f"  [Backtest Sim] Signal BUY at {current_price:.2f}, but cash ({cash:.2f}) is less than price.")


    # End of loop, handle final position if still holding shares (using the last available price)
    if shares > 0 and len(prices) > 0: # prices should have at least one element
        final_day_date = dates.iloc[-1] # Last date from the dates series used in loop or overall
        final_day_price = float(prices.iloc[-1]) # Last price from the prices series
        logger.info(f"  [Backtest Sim] End of simulation period. Selling remaining {shares} shares at final price {final_day_price:.2f} on {final_day_date.date()}.")
        
        fee_final = max(BACKTEST_PARAMS['minimum_fee'], shares * BACKTEST_PARAMS['buy_sell_fee_per_share']) #
        profit_before_tax_final = (final_day_price - last_buy_price) * shares
        tax_final = BACKTEST_PARAMS['tax_rate'] * max(0, profit_before_tax_final) #
        
        cash += shares * final_day_price - fee_final - tax_final
        
        trades_log.append({
            'Date': final_day_date.isoformat(), 'Type': 'SELL_EOD', 
            'Price': final_day_price, 'Shares': shares, 'Cash': cash, 'Portfolio': cash,
            'Reason': 'END_OF_BACKTEST', 'Fee': fee_final, 'Tax': tax_final, 'Profit_Before_Tax': profit_before_tax_final
        })
        shares = 0 

    # Calculate performance statistics
    initial_balance = BACKTEST_PARAMS.get('initial_balance', 10000) #
    final_portfolio_value = cash # Since all shares are sold
    net_profit_pct = (final_portfolio_value / initial_balance - 1) * 100
    
    ticker_change_pct = 0.0
    if len(prices) >= 2: # Need at least two prices to calculate ticker change
        # Ensure prices.iloc[0] is not zero to avoid DivisionByZeroError
        if prices.iloc[0] != 0:
            ticker_change_pct = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
        else:
            logger.warning("⚠️ [Backtest Sim] Initial price is 0, cannot calculate ticker_change_pct.")


    stats = {
        'ticker_change_pct': ticker_change_pct,
        'net_profit_pct': net_profit_pct,
        'final_portfolio_value': final_portfolio_value,
        'initial_balance': initial_balance,
        'total_trades': len(trades_log) # Could be further broken down by buy/sell
    }
    logger.info("✅ [Backtest Sim] Trade simulation done in %.2f s. Stats: %s", time.time() - start_sim_time, stats)
    return trades_log, stats


def backtest_model(request_data: Any) -> Dict[str, Any]: # Changed request_data type to Any for broader compatibility
    """Run full backtest: load, prepare, infer, simulate with robust timeouts."""
    ticker = request_data.stock_ticker
    logger.info("▶️ [Backtest] Initiating backtest for ticker: %s", ticker)
    overall_start_time = time.time()

    try:
        # 1. Load deep learning model and associated data
        logger.info("  [Backtest Step 1/5] Loading deep base model...")
        deep_model, scaler, feature_cols, seq_len = _load_deep_base(ticker)
        logger.info("  ✅ [Backtest Step 1/5] Deep base model loaded.")

        # 2. Load meta-model pipeline
        logger.info("  [Backtest Step 2/5] Loading meta-model pipeline...")
        meta_base_models, meta_final_classifier = _get_meta_pipeline() # Renamed for clarity
        if not meta_final_classifier: # Check if meta_final_classifier is None (problem during load)
            logger.warning("  ⚠️ [Backtest Step 2/5] Meta-model (final classifier) could not be loaded or is None. Proceeding without meta-model.")
            # meta_base_models might also be empty or None, handled by _simulate_trades
        else:
            logger.info("  ✅ [Backtest Step 2/5] Meta-model pipeline loaded.")

        # 3. Prepare data for inference
        logger.info("  [Backtest Step 3/5] Preparing data...")
        # Ensure feature_cols from deep_model loading is used here
        dates, prices, X_scaled = _prepare_data(request_data, feature_cols, scaler, seq_len)
        logger.info("  ✅ [Backtest Step 3/5] Data prepared. X_scaled shape: %s, Dates len: %d", X_scaled.shape, len(dates))

        # 4. Perform inference with the deep learning model
        logger.info("  [Backtest Step 4/5] Performing deep model inference...")
        deep_model_predictions = _infer_deep(deep_model, X_scaled) # Renamed for clarity
        logger.info("  ✅ [Backtest Step 4/5] Deep model inference complete. Predictions shape: %s", deep_model_predictions.shape)

        # 5. Simulate trades
        logger.info("  [Backtest Step 5/5] Simulating trades...")
        trades_history, performance_stats = _simulate_trades(
            dates, prices, deep_model_predictions, meta_base_models, meta_final_classifier
        )
        logger.info("  ✅ [Backtest Step 5/5] Trade simulation complete.")

        total_backtest_time = time.time() - overall_start_time
        logger.info("🏁 [Backtest] Completed successfully for %s in %.2f s.", ticker, total_backtest_time)
        
        final_result = {**performance_stats, 'trades_signals': trades_history}
        logger.info("Final Backtest Stats: %s", performance_stats)
        logger.info("Number of trades: %d", len(trades_history))
        return final_result

    except FileNotFoundError as e:
        logger.error(f"❌ [Backtest] CRITICAL FileNotFoundError during backtest for {ticker}: {e}", exc_info=True)
        return {"error": f"FileNotFoundError: {str(e)}", "trades_signals": [], "net_profit_pct": 0, "ticker_change_pct":0}
    except ValueError as e: # Catch ValueErrors from data prep etc.
        logger.error(f"❌ [Backtest] CRITICAL ValueError during backtest for {ticker}: {e}", exc_info=True)
        return {"error": f"ValueError: {str(e)}", "trades_signals": [], "net_profit_pct": 0, "ticker_change_pct":0}
    except KeyError as e: # Catch KeyErrors from missing columns etc.
        logger.error(f"❌ [Backtest] CRITICAL KeyError during backtest for {ticker}: {e}", exc_info=True)
        return {"error": f"KeyError: {str(e)}", "trades_signals": [], "net_profit_pct": 0, "ticker_change_pct":0}
    except Exception as e:
        logger.error(f"❌ [Backtest] CRITICAL Unhandled exception during backtest for {ticker}: {e}", exc_info=True)
        return {"error": f"Unhandled Exception: {str(e)}", "trades_signals": [], "net_profit_pct": 0, "ticker_change_pct":0}