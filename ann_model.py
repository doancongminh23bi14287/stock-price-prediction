import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from utils import plot_predictions

def run_ann(file_path='data/AAPL.csv'):
    df = pd.read_csv(file_path)
    df['Prev_Close'] = df['Close'].shift(1)
    df.dropna(inplace=True)

    X = df[['Prev_Close']]
    y = df['Close']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, shuffle=False)

    model = Sequential([
        Dense(64, activation='relu', input_dim=1),
        Dense(64, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=16)

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_actual = scaler.inverse_transform(y_test)

    plot_predictions(y_actual, predictions, "ANN Prediction")
    print("MSE:", mean_squared_error(y_actual, predictions))

if __name__ == "__main__":
    run_ann()
