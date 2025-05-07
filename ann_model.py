import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from utils import plot_predictions

def run_ann(file_path='data/AAPL.csv'):
    df = pd.read_csv(file_path)
    df['Prev_Close'] = df['AAPL.Close'].shift(1)
    df.dropna(inplace=True)

    X = df[['Prev_Close']]
    y = df['AAPL.Close']


    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, shuffle=False)

    model = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu', max_iter=500)
    model.fit(X_train, y_train)
    predictions_scaled = model.predict(X_test)

    # Inverse transform predictions to original scale
    predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1))
    y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1))

    plot_predictions(y_test_actual, predictions, "ANN (Sklearn MLPRegressor)")
    print("MSE:", mean_squared_error(y_test_actual, predictions))

if __name__ == "__main__":
    run_ann()
