from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

def get_model():
    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    return model

def main():
    model = get_model()
    print(model.predict([[5.1, 3.5, 1.4, 0.2]]))