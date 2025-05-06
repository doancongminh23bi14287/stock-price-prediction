import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from utils import plot_predictions

def run_rf(file_path='data/AAPL.csv'):
    df = pd.read_csv(file_path)
    df = df[['Close']]

    df['Prev_Close'] = df['Close'].shift(1)
    df.dropna(inplace=True)

    X = df[['Prev_Close']]
    y = df['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    plot_predictions(y_test.values, pred, "Random Forest Prediction")
    print("MSE:", mean_squared_error(y_test, pred))

if __name__ == "__main__":
    run_rf()
