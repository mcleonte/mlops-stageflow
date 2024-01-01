"""
Test inference module for cloud environment,
when ENV_MODE from .env is set to staging or prod.
"""

import pandas as pd
from dotenv import dotenv_values
from google.cloud import aiplatform

ENVS = dotenv_values()


def main():

  aiplatform.init(
      project=ENVS["PROJECT_ID"],
      location=ENVS["REGION"],
  )

  endpoint = aiplatform.Endpoint.list(
      filter=f"display_name={ENVS['ENDPOINT_NAME']}")[0]

  test_path = f'.{ENVS["DATA_PATH"]}/test.csv'
  texts = pd.read_csv(test_path)["narrative"]

  # emulate online prediction with one sample per request
  for text in texts[:10]:
    try:
      response = endpoint.predict(instances=[text])
      print(response[0][0])
    except:
      print("-----------")


if __name__ == "__main__":
  main()
