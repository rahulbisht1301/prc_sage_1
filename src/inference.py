import joblib
import json
import numpy as np

def model_fn(model_dir):
    return joblib.load(f"{model_dir}/model.joblib")

def input_fn(request_body, content_type):
    data = json.loads(request_body)
    return np.array(data["x"]).reshape(-1, 1)

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction, accept):
    return json.dumps({"prediction": prediction.tolist()})
