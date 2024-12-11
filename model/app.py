import numpy as np
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import uvicorn
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Define the request model
class StockPredictionRequest(BaseModel):
    ticker: str
    date: str

# Create FastAPI instance
app = FastAPI()

# Load the trained LSTM model and scaler when the application starts
try:
    model_path = 'model/lstm_model.h5'
    model = load_model(model_path)

    scaler_path = 'model/scaler.pkl'
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load model or scaler: {e}")

# Function to calculate technical indicators
def calculate_technical_indicators(df):
    df['20EMA'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['50SMA'] = df['Close'].rolling(window=50).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df = df.round(2)
    return df

# Function to download stock data and calculate indicators
def download_stock_data(ticker, start_date="2012-01-01"):
    stock_data = yf.download(ticker, start=start_date)
    stock_data['company_name'] = ticker
    stock_data = calculate_technical_indicators(stock_data)
    stock_data.reset_index(inplace=True)  # Reset index to have 'Date' as a column
    return stock_data

# Define the prediction endpoint
@app.post("/predict/")
async def predict(request: StockPredictionRequest):
    ticker = request.ticker
    input_date = request.date

    try:
        # Convert input date to naive datetime
        target_date = pd.to_datetime(input_date).date()  # Use Pandas to ensure compatibility

        # Debugging statement to log the target and current dates
        print(f"Target date: {target_date}, Current date: {datetime.now().date()}")

        # Check if the requested date is in the future
        if target_date > datetime.now().date():  # Compare only date without time
            raise HTTPException(status_code=400, detail="Cannot predict a future date.")

        # Calculate the date range for historical data retrieval (e.g., last 60 days)
        start_date = (target_date - timedelta(days=60)).strftime('%Y-%m-%d')

        # Download historical stock data
        historical_data = download_stock_data(ticker, start_date)

        # Print the downloaded historical data for debugging
        print(historical_data[['Date', 'Adj Close']].head())
        print(f"Last available date in historical data: {historical_data['Date'].iloc[-1].date()}")

        # Check if we have enough data to predict the requested date
        if historical_data.empty or historical_data['Date'].iloc[-1].date() < target_date:
            print("Insufficient data for the prediction.")
            raise HTTPException(status_code=404, detail="Insufficient data to predict the requested date.")

        # Prepare the last 60 adjusted close prices for prediction
        historical_prices = historical_data['Adj Close'].values[-60:]  # Get the last 60 adjusted close prices
        input_features = np.array(historical_prices).reshape(-1, 1)  # Reshape to (60, 1)

        # Scale the historical data
        scaled_features = scaler.transform(input_features)

        # Reshape for LSTM input: (1, 60, 1)
        lstm_input = scaled_features.reshape(1, 60, 1)

        # Make a prediction
        scaled_prediction = model.predict(lstm_input)[0][0]

        # Reverse the scaling to get the original predicted price
        predicted_price = scaler.inverse_transform([[scaled_prediction]])[0][0]

        # Return the predicted price along with the date
        return {"date": input_date, "predicted_price": predicted_price}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# To run the FastAPI app, use the command:
# uvicorn app:app --reload

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
