import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error

model = joblib.load('/opt/ml/model/model.joblib')
data = pd.read_csv('/opt/ml/input/data/train/data.csv')

X = data[['x']]
y = data['y']

preds = model.predict(X)
mse = mean_squared_error(y, preds)

print(f"MSE: {mse}")

# Fail pipeline if performance is bad
if mse > 1:
    raise Exception("Model performance is not acceptable")
