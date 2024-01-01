"""
References:
https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements#prediction
https://cloud.google.com/vertex-ai/docs/predictions/use-custom-container#aiplatform_upload_model_highlight_container-python_vertex_ai_sdk
https://cloud.google.com/vertex-ai/docs/predictions/custom-prediction-routines
"""

import os
import json

import joblib
from flask import Flask, request, make_response
import numpy as np
import tensorflow as tf

ENV_MODE = os.environ["ENV_MODE"]
MODEL_VERSION_ALIAS = os.environ["MODEL_VERSION_ALIAS"]
BATCH_SIZE = os.environ["BATCH_SIZE"]
# PREPROCESSORS_PATH = os.environ["PREPROCESSORS_PATH"]

PORT = os.environ["PORT"]
PREDICT_ROUTE = os.environ["PREDICT_ROUTE"]
EVAL_ROUTE = os.environ["EVAL_ROUTE"]
HEALTH_ROUTE = os.environ["HEALTH_ROUTE"]

# preload preprocessors
label_encoder = joblib.load("preprocessors/label_encoder.joblib")
vectorizer = joblib.load("preprocessors/vectorizer.joblib")
selector = joblib.load("preprocessors/selector.joblib")

# preload model
model = tf.keras.models.load_model(f"models/{MODEL_VERSION_ALIAS}/model")

app = Flask(__name__)


@app.route(PREDICT_ROUTE, methods=["POST"])
def predict():

  x = request.json["instances"]

  # embed request
  x = vectorizer.transform(x)
  x = selector.transform(x).astype("float32").toarray()

  # make prediction
  y_pred = model(x, training=False)

  # get predicted classes from probability distribution vectors
  y_pred = np.argmax(y_pred, axis=-1, keepdims=True).reshape(-1)

  # get label name from encoded label number
  y_pred = label_encoder.inverse_transform(y_pred)

  # return response with predicitons as serialised list
  response = json.dumps({"predictions": y_pred.tolist()})

  return response


@app.route(EVAL_ROUTE, methods=["POST"])
def evaluate():

  xy = request.json
  x = xy["narrative"]
  y = xy["product"]

  results = model.evaluate(
      x,
      y,
      batch_size=BATCH_SIZE,
  )

  return json.dumps({"eval_acc": results[1], "eval_loss": results[0]})


@app.route(HEALTH_ROUTE, methods=["GET"])
def healthcheck():
  return make_response("OK", 200)


if __name__ == "__main__":

  app.run(host="0.0.0.0", port=PORT, debug=ENV_MODE == "debug")
