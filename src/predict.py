import pickle
import numpy as np

def load_model():
    model = pickle.load(open("models/random_forest.pkl", "rb"))
    scaler = pickle.load(open("models/scaler.pkl", "rb"))
    return model, scaler

def predict(transaction_features):
    model, scaler = load_model()

    transaction_features = np.array(transaction_features).reshape(1, -1)
    transaction_scaled = scaler.transform(transaction_features)

    prediction = model.predict(transaction_scaled)

    return prediction[0]

if __name__ == "__main__":
    # Example dummy input (replace with real 30 feature values)
    sample = [0]*30
    result = predict(sample)

    if result == 1:
        print("Fraudulent Transaction Detected!")
    else:
        print("Legitimate Transaction")