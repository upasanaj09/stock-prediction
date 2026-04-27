from flask import Flask,render_template, request
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_stock', methods=['POST'])
def get_stock():
    symbol = request.form['symbol']
    
    try:
        # Fetch stock data
        stock = yf.Ticker(symbol)
        data = stock.history(period="6mo")

        if data.empty:
            return render_template('index.html', error="Invalid stock symbol")
    
    except Exception:
        return render_template('index.html', error="Error fetching data")

    data = data.reset_index()
    data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%Y-%m-%d')
    
    #Calculate RSI

    delta=data['Close'].diff()

    gain=delta.clip(lower=0)
    loss=-delta.clip(upper=0)

    avg_gain = gain.ewm(com=13,adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(com=13,adjust=False, min_periods=14).mean()

    rs = avg_gain / (avg_loss + 1e-10)

    data['RSI']=(100-(100/(1+rs))).clip(0,100)

    # --- Features ---
    data['MA5']       = data['Close'].rolling(5).mean()
    data['MA10']      = data['Close'].rolling(10).mean()
    data['MA20']      = data['Close'].rolling(20).mean()
    data['Lag1']      = data['Close'].shift(1)
    data['Lag2']      = data['Close'].shift(2)
    data['Lag5']      = data['Close'].shift(5)
    data['Volatility']= data['Close'].rolling(5).std()
    data['MA_cross']  = data['MA5'] - data['MA20']

    data = data.dropna().copy()

    features = ['MA5', 'MA10', 'MA20', 'Lag1', 'Lag2', 'Lag5', 'Volatility', 'MA_cross', 'RSI']
    X = data[features]
    y = data['Close']

    # --- Train/test split (no leakage) ---
    split = int(len(X) * 0.8)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X.iloc[:split], y.iloc[:split])

    # Predictions only on test portion
    predicted_prices = [None] * split + model.predict(X.iloc[split:]).tolist()

    # --- Next day prediction ---
    next_day = pd.DataFrame([{
        'MA5':        data['MA5'].iloc[-1],
        'MA10':       data['MA10'].iloc[-1],
        'MA20':       data['MA20'].iloc[-1],
        'Lag1':       data['Close'].iloc[-1],
        'Lag2':       data['Close'].iloc[-2],
        'Lag5':       data['Close'].iloc[-5],
        'Volatility': data['Volatility'].iloc[-1],
        'MA_cross':   data['MA_cross'].iloc[-1],
        'RSI':        data['RSI'].iloc[-1]
    }])
    predicted_price = model.predict(next_day)[0]

    # Round values
    data['Close'] = data['Close'].round(2)
    data['MA5'] = data['MA5'].round(2)
    data['RSI'] = data['RSI'].round(2)
    
    #Format for display
    dates = data['Date'].tolist()
    prices = data['Close'].tolist()
    ma5=data['MA5'].tolist()
    rsi = data['RSI'].tolist()
    
    #Add next day prediction
    dates.append("Next Day")
    prices.append(round(predicted_price,2))
    predicted_prices.append(predicted_price)
    ma5.append(None)
    rsi.append(None)

    #Buy/Sell logic
    # Get only valid RSI values (remove None)
    valid_rsi = [x for x in rsi if x is not None]

    # Safe latest RSI   
    if valid_rsi:
        latest_rsi = valid_rsi[-1]
    else:
        latest_rsi = 50   # default safe value
    
    if latest_rsi<30:
        signal = "BUY"
    elif latest_rsi>70:
        signal="SELL"
    else:
        signal = "HOLD"

    #Table Data
    table_data = data.to_dict(orient='records')

    return render_template(
        'result.html',
        symbol=symbol,
        table_data=table_data,
        dates=dates,
        prices=prices,
        prediction=round(predicted_price,2),
        predicted_prices=predicted_prices,
        ma5=ma5,
        rsi=rsi,
        signal=signal
    )


if __name__=='__main__':
    app.run(debug=True)