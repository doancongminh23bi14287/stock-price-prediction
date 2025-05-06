import matplotlib.pyplot as plt

def plot_predictions(actual, predicted, title):
    plt.figure(figsize=(10, 5))
    plt.plot(actual, label="Actual")
    plt.plot(predicted, label="Predicted")
    plt.title(title)
    plt.legend()
    plt.show()
