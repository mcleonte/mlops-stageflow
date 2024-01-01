import os

import joblib
import pandas as pd
import tensorflow as tf

from . import PATHS, ENVS
from .data import main as data_main
from .train import main as train_main


def ensure_test_data():

    if not all(
        os.path.exists(PATHS[path])
            for path in (
                "test",
                "label_encoder",
                "vectorizer",
                "selector"
            )
    ):

        print("Missing test data, running data module...")
        data_main()


def ensure_model():
    
    if not os.path.exists(PATHS["model"]) or not os.listdir(PATHS["model"]):
        print(f'Model not found at path {PATHS["model"]},running train module...') 
        model, _ = train_main()
    else:
        model = tf.keras.models.load_model(PATHS["model"])
    return model


def main():

    ensure_test_data()

    # load test data and resources
    test = pd.read_csv(PATHS["test"], engine="pyarrow")
    label_encoder = joblib.load(PATHS["label_encoder"])
    vectorizer = joblib.load(PATHS["vectorizer"])
    selector = joblib.load(PATHS["selector"])

    x_test = test["narrative"]
    y_test = test["product"]

    # embed texts
    x_test = vectorizer.transform(x_test)
    x_test = selector.transform(x_test).astype('float32')

    # encode labels
    y_test = label_encoder.transform(y_test)

    model = ensure_model()

    results = model.evaluate(
        x_test,
        y_test,
        batch_size=int(ENVS["BATCH_SIZE"]),
    )

    print(f"Evaluation - accuracy: {results[1]}, loss: {results[0]}")


if __name__=="__main__":
    main()