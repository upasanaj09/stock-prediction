from flask import Flask,render_template, request
import numpy as np
import pandas as pd
import yfinance as yf
import keras
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

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
        data = stock.history(period="1y")

        if data.empty:
            return render_template('index.html', error=f"No data found for '{symbol}'. "
                                   "It may be delisted or invalid. "
                                   "For NSE stocks use format: RELIANCE.NS")
    
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

    #Moving average
    data['MA5'] = data['Close'].rolling(5).mean().bfill()
    data['MA10']       = data['Close'].rolling(10).mean()
    data['Volatility'] = data['Close'].rolling(5).std()
    data = data.dropna().copy()

    features = ['Close', 'MA5', 'MA10', 'RSI', 'Volatility']

    #Scale Close prices
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data[features])  

    close_scaler = MinMaxScaler()
    close_scaler.fit_transform(data[['Close']])

    #Build sequences
    SEQ_LEN = 15
    X_seq, y_seq = [], []
    for i in range(SEQ_LEN, len(scaled)):
        X_seq.append(scaled[i - SEQ_LEN:i])   
        y_seq.append(scaled[i,0]) # next day price

    X_seq = np.array(X_seq)  
    y_seq = np.array(y_seq)   

    #Train/ test split
    split    = int(len(X_seq) * 0.8)
    X_train  = X_seq[:split]
    X_test   = X_seq[split:]
    y_train  = y_seq[:split]
    y_test   = y_seq[split:]

    #Build LSTM 
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(SEQ_LEN, len(features))),
        Dropout(0.2),
        LSTM(64,return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.1,
        verbose=0             
    )

    #Predict on test set
    test_preds_scaled = model.predict(X_test, verbose=0)

    # Inverse transform → back to real prices
    test_preds = close_scaler.inverse_transform(test_preds_scaled).flatten()
    y_actual   = close_scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

    #Align predicted_prices list with full date range
    n_none           = SEQ_LEN + split
    predicted_prices = [None] * n_none + test_preds.tolist()

    #Next day prediction
    last_seq       = scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, len(features))
    next_scaled    = model.predict(last_seq, verbose=0)
    predicted_price = float(close_scaler.inverse_transform(next_scaled)[0][0])

   #Evaluation 
    def directional_accuracy(actual, pred):
        a_dir = np.diff(actual) > 0
        p_dir = np.diff(pred)  > 0
        return round(float(np.mean(a_dir == p_dir) * 100), 2)

    def tolerance_accuracy(actual, pred, tol=0.02):
        errors = np.abs(actual - pred) / actual
        return round(float(np.mean(errors <= tol) * 100), 2)

    rmse          = round(float(np.sqrt(mean_squared_error(y_actual, test_preds))), 2)
    rmse_percent  = round(float(rmse / y_actual.mean() * 100), 2)
    mape          = round(float(np.mean(np.abs((y_actual - test_preds) / y_actual)) * 100), 2)
    dir_acc       = directional_accuracy(y_actual, test_preds)
    tol_acc       = tolerance_accuracy(y_actual, test_preds)

    evaluation = {
        'rmse':                  rmse,
        'rmse_percent':          rmse_percent,
        'mape':                  mape,
        'directional_accuracy':  dir_acc,
        'tolerance_accuracy':    tol_acc,
        'total_test_days':       len(y_actual)
    }

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

   #Signal 
    valid_rsi  = [x for x in rsi if x is not None]
    latest_rsi = valid_rsi[-1] if valid_rsi else 50
    signal     = "BUY" if latest_rsi < 30 else ("SELL" if latest_rsi > 70 else "HOLD")

    return render_template(
        'result_demo.html',
        symbol=symbol,
        table_data=data.to_dict(orient='records'),
        dates=dates,
        prices=prices,
        prediction=round(predicted_price,2),
        predicted_prices=predicted_prices,
        ma5=ma5,
        rsi=rsi,
        signal=signal,
        evaluation=evaluation
    )


if __name__=='__main__':
    app.run(debug=True)
