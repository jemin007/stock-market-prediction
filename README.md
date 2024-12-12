# Stock Price Prediction API - README

This project is a FastAPI-based web application for predicting stock prices using a trained LSTM model. The model uses historical stock data and other features to forecast the next stock price.

## Features
- **FastAPI Framework**: Provides a lightweight, high-performance API for model inference.
- **LSTM Model**: Leverages a trained Long Short-Term Memory (LSTM) model for time-series prediction.
- **Scalable and Modular**: Easily extendable to incorporate more features or models.
- **Automatic Data Scaling**: Uses a pre-fitted scaler to process input data.

## Requirements
- **Python**: 3.8 or above  
- **Required Libraries**:
  - `fastapi`
  - `uvicorn`
  - `tensorflow`
  - `numpy`
  - `pandas`
  - `scikit-learn`

## Installation
1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-repo/stock-price-prediction-api.git
   cd stock-price-prediction-api
   ```
2. **Create a Virtual Environment**
   ```bash
   python -m venv test
   source venv/bin/activate  # On Linux/Mac
   test\Scripts\activate     # On Windows
   ```
3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure Required Files Exist**:
   - `lstm_model.h5`: Trained LSTM model file.
   - `scaler.pkl`: Pre-fitted scaler file for data normalization.

## Usage
### Run the API
Start the server using Uvicorn:
```bash
uvicorn app:app --reload
```
The API will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000).

### API Endpoints
#### `POST /predict/`
Predicts the stock price based on input features.

**Request Body**:
```json
{
  "adj_close": 152.0,
  "open": 150.0,
  "high": 155.0,
  "low": 148.0,
  "close": 152.0,
  "volume": 100000,
  "ema20": 150.5,
  "sma50": 151.0,
  "rsi": 60.0,
  "vix": 22.0
}
```

**Response**:
```json
{
  "predicted_price": 153.5
}
```

**Error Handling**:  
Returns `500 Internal Server Error` for any prediction errors.

### Files in the Project
- `app.py`: Main application file containing the FastAPI code.
- `lstm_model.h5`: Pre-trained LSTM model file.
- `scaler.pkl`: Pre-fitted scaler for normalizing input data.
- `requirements.txt`: List of dependencies required for the project.

### Testing
Use tools like Postman or `curl` to test the `/predict/` endpoint.

**Example using curl**:
```bash
curl -X POST "http://127.0.0.1:8000/predict/" \
-H "Content-Type: application/json" \
-d '{
  "adj_close": 152.0,
  "open": 150.0,
  "high": 155.0,
  "low": 148.0,
  "close": 152.0,
  "volume": 100000,
  "ema20": 150.5,
  "sma50": 151.0,
  "rsi": 60.0,
  "vix": 22.0
}'
```

## Future Enhancements
- Add more advanced error handling and logging.
- Integrate support for multiple stock prediction models.
- Enhance input validation and feature extraction.
