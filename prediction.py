import torch
from model_training import AttentionModel
from config import FEATURE_COLUMNS
from data_processing import get_data


def predict_model(ticker, start_date):
    # Load the model
    model = AttentionModel(input_size=len(FEATURE_COLUMNS), hidden_size=64, output_size=1)
    model.load_state_dict(torch.load(f"models/trained_model-{ticker}.pkl"))
    model.eval()

    # Fetch the data
    df = get_data(ticker, start_date, pd.Timestamp.today().strftime("%Y-%m-%d"))

    # Prepare features
    X = df[FEATURE_COLUMNS].values
    X = torch.tensor(X, dtype=torch.float32)

    # Make prediction
    with torch.no_grad():
        prediction = model(X)

    # The output is the predicted price change or direction
    predicted_change = prediction.numpy()

    signal = "BUY" if predicted_change > 0 else "SELL"

    return signal
