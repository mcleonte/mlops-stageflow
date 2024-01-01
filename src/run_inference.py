"""
Run inference module for cloud environment,
when ENV_MODE from .env is set to staging or prod.
"""

from dotenv import dotenv_values
from google.cloud import aiplatform

ENVS = dotenv_values()


def ensure_endpoint(display_name: str,):
  """
    Creates an endpoint if it does not exist yet.
    """

  endpoint = aiplatform.Endpoint.list(
      filter=f"display_name={ENVS['ENDPOINT_NAME']}")

  if endpoint:
    return endpoint[0]

  print("Creating new endpoint...")

  endpoint = aiplatform.Endpoint.create(display_name=display_name,)

  return endpoint


def main():

  aiplatform.init(
      project=ENVS["PROJECT_ID"],
      location=ENVS["REGION"],
  )

  endpoint = ensure_endpoint(display_name=ENVS["ENDPOINT_NAME"],)

  model = aiplatform.Model(
      model_name=ENVS["MODEL_ID"],
      version=ENVS["MODEL_VERSION_ALIAS"],
  )

  machine_type = \
      ENVS[f'{ENVS["ENV_MODE"].upper()}_INFERENCE_MACHINE_TYPE']

  deployed_model_display_name = \
      f'{ENVS["MODEL_ID"]}-{ENVS["MODEL_VERSION_NUMBER"]}'

  print("Deploying model to endpoint...")
  model.deploy(
      endpoint=endpoint,
      deployed_model_display_name=deployed_model_display_name,
      # traffic_percentage=traffic_percentage,
      # traffic_split=traffic_split,
      machine_type=machine_type,
      # min_replica_count=min_replica_count,
      # max_replica_count=max_replica_count,
      # accelerator_type=accelerator_type,
      # accelerator_count=accelerator_count,
      # explanation_metadata=explanation_metadata,
      # explanation_parameters=explanation_parameters,
      # metadata=metadata,
      # sync=sync,
  )

  model.wait()

  return model


if __name__ == "__main__":
  main()
