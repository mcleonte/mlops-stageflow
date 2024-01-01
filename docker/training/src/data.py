# https://developers.google.com/machine-learning/guides/text-classification/step-3

from typing import Tuple
import os

import joblib
import kaggle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder

from . import ENVS, PATHS


def split_dataset(
    raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """
    Splits the raw dataset into separate
    train, validation and test sets.
    """

  train_val, test = train_test_split(
      raw, train_size=0.8, shuffle=True, random_state=int(ENVS["SPLIT_SEED"]))

  train, val = train_test_split(
      train_val,
      train_size=0.75,  # 0.8 * 0.25 = 0.2
      shuffle=True,
      random_state=int(ENVS["SPLIT_SEED"]))

  return train, val, test


def ngram_vectorize(
    train_texts: list[str],
    train_labels: np.ndarray,
    val_texts: list[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Vectorizes texts as n-gram vectors.
    1 text = 1 tf-idf vector the length of vocabulary of unigrams + bigrams.
    """

  # create keyword arguments to pass to the 'tf-idf' vectorizer.
  kwargs = {
      'ngram_range': eval(ENVS["NGRAM_RANGE"]),  # Use 1-grams + 2-grams.
      'dtype': 'int32',
      'strip_accents': 'unicode',
      'decode_error': 'replace',
      'analyzer': ENVS["TOKEN_MODE"],  # Split text into word tokens.
      'min_df': int(ENVS["MIN_DOCUMENT_FREQUENCY"]),
      'stop_words': 'english',
  }
  vectorizer = TfidfVectorizer(**kwargs)

  # learn vocabulary from training texts and vectorize training texts.
  x_train = vectorizer.fit_transform(train_texts)

  # vectorize validation and test texts.
  x_val = vectorizer.transform(val_texts)

  # select top 'k' of the vectorized features.
  selector = SelectKBest(
      f_classif,
      k=min(int(ENVS["TOP_K"]), x_train.shape[1]),
  )
  selector.fit(x_train, train_labels)

  x_train = selector.transform(x_train).astype('float32')
  x_val = selector.transform(x_val).astype('float32')

  return x_train, x_val, vectorizer, selector


def main():

  # ensure raw data

  if not os.path.isfile(PATHS["raw"]):

    kaggle.api.authenticate()

    kaggle.api.dataset_download_files(
        ENVS["KAGGLE_DATASET"],
        path=PATHS["data"],
        unzip=True,
    )

  raw = pd.read_csv(PATHS["raw"], engine="pyarrow", index_col=0)

  # ensure cleaned data

  try:

    cleaned = pd.read_csv(PATHS["cleaned"], engine="pyarrow")

  except FileNotFoundError:

    raw.dropna(inplace=True)
    raw.drop_duplicates(inplace=True)

    # placeholder for other data cleaning operations
    cleaned = raw

    cleaned.to_csv(PATHS["cleaned"], index=False)

  # split data and save test set for evaluation step

  train, val, test = split_dataset(cleaned)
  test.to_csv(PATHS["test"], index=False)

  # ensure label encodings

  if not all(
      os.path.isfile(PATHS[path])
      for path in ("label_encoder", "y_train", "x_train")):

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train["product"])
    y_val = label_encoder.transform(val["product"])

    joblib.dump(label_encoder, PATHS["label_encoder"])
    joblib.dump(y_train, PATHS["y_train"])
    joblib.dump(y_val, PATHS["y_val"])

  else:

    label_encoder = joblib.load(PATHS["label_encoder"])
    y_train = joblib.load(PATHS["y_train"])
    y_val = joblib.load(PATHS["y_val"])

  # ensure text embeddings

  try:

    vectorizer = joblib.load(PATHS["vectorizer"])
    selector = joblib.load(PATHS["selector"])
    x_train = joblib.load(PATHS["x_train"])
    x_val = joblib.load(PATHS["x_val"])

  except FileNotFoundError:

    x_train, x_val, vectorizer, selector = \
        ngram_vectorize(
            train["narrative"],
            train["product"],
            val["narrative"],
        )

    joblib.dump(vectorizer, PATHS["vectorizer"])
    joblib.dump(selector, PATHS["selector"])
    joblib.dump(x_train, PATHS["x_train"])
    joblib.dump(x_val, PATHS["x_val"])


if __name__ == "__main__":
  main()
