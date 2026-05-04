# AI Stock Prediction System

A machine learning web application that predicts next-day stock prices for NSE/BSE listed Indian stocks using LSTM deep learning, technical indicators, and sentiment analysis.

Built with Flask, TensorFlow/Keras, and Chart.js.

---

## ✨ Features

- **LSTM Price Prediction** — Deep learning model trained on 1 year of historical data
- **Technical Indicators** — RSI, MA5, MA10, Bollinger Bands, Volatility, Volume Ratio
- **Buy / Sell / Hold Signal** — Based on RSI overbought/oversold zones
- **Model Evaluation** — RMSE, MAPE, Directional Accuracy, Tolerance Accuracy
- **Dark Terminal UI** — dark themed dashboard

---

## 🖥️ Screenshots
<img width="1917" height="892" alt="image" src="https://github.com/user-attachments/assets/0fa9d0fb-8ae8-4721-95e8-f286d8e2a55d" />
<img width="1890" height="904" alt="image" src="https://github.com/user-attachments/assets/f75333fd-43cb-4a7f-8817-6cd0b6420c4a" />

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| ML Model | LSTM (TensorFlow / Keras) |
| Data Source | yfinance (Yahoo Finance) |
| Technical Analysis | pandas, numpy, scikit-learn |
| Frontend | HTML, CSS, Chart.js |
| Deployment | Render |

---

## 📁 Project Structure

```
stock-prediction/
├── app.py                  # Flask routes and LSTM logic
├── templates/
│   ├── index.html          # Home page with ticker and search
│   └── result.html         # Dashboard with chart and evaluation
├── static/
│   └── style.css           # Additional styles (if any)
├── requirements.txt        # Python dependencies
├── runtime.txt             # Python version
└── README.md
```

---

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/upasanaj09/stock-prediction.git
cd stock-prediction
```

**2. Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
python app.py
```

**5. Open in browser**
```
http://127.0.0.1:5000
```

---

## 📦 Requirements

```
flask
yfinance
pandas
numpy
scikit-learn
tensorflow
keras
gunicorn
```

---

## 🧠 How the Model Works

```
1. Fetch 1 year of historical OHLCV data via yfinance
2. Calculate technical features:
       RSI (Wilder's smoothing, period=14)
       MA5, MA10 (rolling averages)
       Volatility (5-day rolling std)
       Volume Ratio (volume / MA volume)
       Bollinger Band Width
       Daily Returns (pct_change)
3. Scale all features with MinMaxScaler
4. Build 10-day sliding window sequences
5. Train 3-layer LSTM (128 → 64 → 32 units)
6. Predict next day closing price
7. Evaluate on 20% held-out test data
```

---

## 📊 Model Evaluation Metrics

| Metric | Description |
|---|---|
| Directional Accuracy | Did price go up/down correctly? |
| Tolerance Accuracy ±2% | Prediction within 2% of actual |
| MAPE | Mean Absolute Percentage Error |
| RMSE | Root Mean Square Error in ₹ |
| RMSE % | RMSE relative to average price |

---

## 📡 Supported Stock Symbols

Use NSE format with `.NS` suffix:

```
RELIANCE.NS    TCS.NS         INFY.NS
HDFCBANK.NS    WIPRO.NS       ICICIBANK.NS
BAJFINANCE.NS  SBIN.NS        TATAMOTORS.NS
```


## ⚠️ Disclaimer

This project is built for **educational purposes** as part of an MCA final year project. Stock predictions are not financial advice. Always consult a certified financial advisor before investing.

---

## 👨‍💻 Author

**Upasana Joshi**
MCA Final Year — College of Technology, GBPUAT, Pantnagar

---

## 📄 License

MIT License — free to use and modify.
