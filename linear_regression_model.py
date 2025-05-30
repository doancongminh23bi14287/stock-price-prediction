import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from utils import plot_predictions

def run_lr(file_path='data/AAPL.csv'):
    df = pd.read_csv(file_path)
    df['Prev_Close'] = df['AAPL.Close'].shift(1)
    df.dropna(inplace=True)

    X = df[['Prev_Close']]
    y = df['AAPL.Close']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    plot_predictions(y_test.values, pred, "Linear Regression Prediction")
    print("MSE:", mean_squared_error(y_test, pred))

if __name__ == "__main__":
    run_lr()
