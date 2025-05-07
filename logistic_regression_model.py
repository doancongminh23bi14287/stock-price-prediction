import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

def run_logistic_regression(file_path='data/AAPL.csv'):
    df = pd.read_csv(file_path)
    df['Target'] = (df['AAPL.Close'].shift(-1) > df['AAPL.Close']).astype(int)  # 1 if price goes up
    df['Prev_Close'] = df['AAPL.Close'].shift(1)
    df.dropna(inplace=True)

    X = df[['Prev_Close']]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    run_logistic_regression()
