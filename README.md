


# Stock Market Prediction ML Project

This project is a Stock Market Predictor application built using Streamlit and a pre-trained LSTM model created with Keras. The application downloads historical stock data using yfinance, visualizes moving averages, and predicts future stock prices.

## Features

- Download historical stock data for a given stock symbol.
- Visualize stock closing prices with moving averages (MA50, MA100, MA200).
- Predict stock prices using a pre-trained LSTM model.
- Interactive web app interface using Streamlit.

## Installation

1. Clone the repository.
2. Install the required Python packages using:

```
pip install -r requirements.txt
```

## Usage

Run the Streamlit app with the following command:

```
streamlit run app.py
```

Enter a stock symbol (e.g., GOOG) in the input box to view stock data, visualizations, and predictions.

## Files

- `app.py`: Streamlit application script.
- `stock_model.h5`: Pre-trained Keras LSTM model.
- `Stock_Market_Prediction_Model_Creation.ipynb`: Jupyter notebook for model creation and training.

## Dependencies

- numpy
- pandas
- matplotlib
- yfinance
- scikit-learn
- keras
- streamlit
- pyplot

## License

This project is open source and free to use.
