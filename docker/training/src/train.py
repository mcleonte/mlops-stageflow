import os

import joblib
import pandas as pd
import tensorflow as tf

from . import PATHS, ENVS
from .model import create_mlp_model
from .data import main as data_main


def train_ngram_model(
    x_train,
    y_train,
    x_val,
    y_val,
    learning_rate=1e-3,
    epochs=1000,
    batch_size=int(ENVS["BATCH_SIZE"]),
    layers=2,
    units=64,
    dropout_rate=0.2,
):
  """
    Trains n-gram model on the given dataset.

    # Arguments
        data: tuples of training and test texts and labels.
        learning_rate: float, learning rate for training model.
        epochs: int, number of epochs.
        batch_size: int, number of samples per batch.
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of Dense layers in the model.
        dropout_rate: float: percentage of input to drop at Dropout layers.

    # Raises
        ValueError: If validation data has label values which were not seen
            in the training data.
    """

  # Verify that validation labels
  # are in the training labels set.
  train_classes = pd.unique(y_train)
  num_classes = len(train_classes)
  unexpected_labels = [y for y in y_val if y not in train_classes]
  if len(unexpected_labels):
    raise ValueError(
        'Unexpected label values found in the validation set:'
        ' {unexpected_labels}. Please make sure that the '
        'labels in the validation set are in the same range '
        'as training labels.'.format(unexpected_labels=unexpected_labels))

  # Create model instance.
  model = create_mlp_model(
      layers=layers,
      units=units,
      dropout_rate=dropout_rate,
      input_shape=x_train.shape[1:],
      num_classes=num_classes,
      learning_rate=learning_rate,
  )

  # create callback for early stopping on validation loss
  # if the loss does not decrease in two consecutive tries,
  # then stop training
  callbacks = tf.keras.callbacks.EarlyStopping(
      monitor='val_loss',
      patience=2,
  )

  # train and validate model
  history = model.fit(
      x_train,
      y_train,
      epochs=epochs,
      callbacks=callbacks,
      validation_data=(x_val, y_val),
      verbose=2,  # logs once per epoch
      batch_size=batch_size,
  )

  # print results
  history = history.history
  print("Validation accuracy: {acc}, loss: {loss}".format(
      acc=history['val_acc'][-1],
      loss=history['val_loss'][-1],
  ))

  # save model
  print(f'Saving model to {PATHS["model"]} ...')
  model.save(PATHS["model"])

  return model, history


def ensure_train_data():

  if not all(
      os.path.exists(PATHS[path])
      for path in ("x_train", "x_val", "y_train", "y_val")):

    print("Missing train data, running data module...")
    data_main()


def main():

  ensure_train_data()

  x_train = joblib.load(PATHS["x_train"]).toarray()
  x_val = joblib.load(PATHS["x_val"]).toarray()
  y_train = joblib.load(PATHS["y_train"])
  y_val = joblib.load(PATHS["y_val"])

  return train_ngram_model(
      x_train,
      y_train,
      x_val,
      y_val,
  )


if __name__ == "__main__":
  main()
