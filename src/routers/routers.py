from fastapi import HTTPException
import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from config import MODEL_PARAMS
from data import data_processing
from models import trainer, backtester
from routers.routers_entities import UpdateIndicatorsData
from visualization.visualization_plot import generate_trade_plot

router = APIRouter(prefix="", tags=["Booking Items"])


@router.post("/indicator_data")
async def get_data(request_data: UpdateIndicatorsData):
    try:
        response = data_processing.get_data(request_data)

        return JSONResponse(response.to_dict(orient="records"), status_code=200)

    except Exception as err:
        logging.exception(err)
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.post("/train-model")
async def train_model(request_data: UpdateIndicatorsData):
    try:
        response = trainer.train_single(request_data)

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
            "buying_threshold": MODEL_PARAMS.get("grid_buying_threshold"),
            "selling_threshold": MODEL_PARAMS.get("grid_selling_threshold"),
            "profit_target": MODEL_PARAMS.get("grid_profit_target"),
            "trailing_stop": MODEL_PARAMS.get("grid_trailing_stop"),
        }
        df = backtester.optimize_signal_params(request_data, param_grid)
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
