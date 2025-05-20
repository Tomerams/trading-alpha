import io
from fastapi import BackgroundTasks, HTTPException
import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from backtest import backtester
from config.optimizations_config import BACKTEST_OPTIMIZATIONS_PARAMS, OPTUNA_PARAMS
from models.model_signals_decision_train import train_meta_model_from_request
from data import data_processing
from backtest import backtest_tuning
from models.model_prediction_tuning import run_optuna
from models import model_prediction_trainer
from routers.routers_entities import UpdateIndicatorsData
from visualization.visualization_plot import generate_trade_plot

router = APIRouter(prefix="", tags=["Booking Items"])


@router.post("/indicators", summary="Fetch computed indicator data for an asset")
async def get_data(request_data: UpdateIndicatorsData):
    try:
        response = data_processing.get_indicators_data(request_data)

        return JSONResponse(response.to_dict(orient="records"), status_code=200)

    except Exception as err:
        logging.exception(err)
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.post("/train", summary="Train a forecasting model on historical data")
async def train_model(request_data: UpdateIndicatorsData):
    try:
        response = model_prediction_trainer.train_single(request_data)

        return JSONResponse(response.to_dict(orient="records"), status_code=200)

    except Exception as err:
        logging.exception(err)
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.post("/backtest")
async def backtest(request_data: UpdateIndicatorsData):
    try:
        response = backtester.backtest_model(request_data)

        return JSONResponse(response, status_code=200)

    except Exception as err:
        logging.exception(err)
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.post("/optimize-signals")
async def optimize_signals(request_data: UpdateIndicatorsData):
    """
    Runs a grid search over trading signal parameters and returns a ranked list of results.
    Expects MODEL_PARAMS to include keys buying_threshold, selling_threshold,
    profit_target, trailing_stop as lists.
    """
    try:
        # param_grid is provided via MODEL_PARAMS or can be sent in request_data
        param_grid = {
            "buying_threshold": BACKTEST_OPTIMIZATIONS_PARAMS.get(
                "grid_buying_threshold"
            ),
            "selling_threshold": BACKTEST_OPTIMIZATIONS_PARAMS.get(
                "grid_selling_threshold"
            ),
            "profit_target": BACKTEST_OPTIMIZATIONS_PARAMS.get("grid_profit_target"),
            "trailing_stop": BACKTEST_OPTIMIZATIONS_PARAMS.get("grid_trailing_stop"),
        }
        df = backtest_tuning.optimize_signal_params(request_data, param_grid)
        # ensure DataFrame to JSON
        return JSONResponse(df.to_dict(orient="records"), status_code=200)
    except Exception as err:
        logging.exception(err)
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.post("/plot-trades", summary="Generate trade signal overlay chart")
async def plot_trades(request_data: UpdateIndicatorsData):
    """
    Delegates plot generation to visualization_service, returning a PNG stream.
    """
    try:
        buf = generate_trade_plot(request_data)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        logging.exception("Error generating trade plot")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tune-hparams", summary="Run hyperparameter optimization")
async def tune_hyperparams(
    request_data: UpdateIndicatorsData, background_tasks: BackgroundTasks
):
    """
    Launch Optuna sweep in the background and return immediately.
    """
    # schedule the long-running job
    background_tasks.add_task(run_optuna, request_data)
    return JSONResponse(
        {"status": "started", "n_trials": OPTUNA_PARAMS["n_trials"]}, status_code=202
    )


@router.get("/tune-hparams/result", summary="Get last tuning results")
def get_tuning_result():
    """
    Load the persisted study and return best params + value.
    """
    import joblib

    study = joblib.load("files/models/optuna_study.pkl")
    return JSONResponse(
        {"best_params": study.best_params, "best_value": study.best_value}
    )


@router.post("/train-meta-model", summary="Train the meta-model (BUY/SELL/HOLD AI)")
async def meta_train_ai(request_data: UpdateIndicatorsData):
    try:
        result = train_meta_model_from_request(request_data)
        return {"status": "success", **result}
    except Exception as err:
        logging.exception("Meta model training failed.")
        raise HTTPException(status_code=500, detail="Meta model training failed.")
