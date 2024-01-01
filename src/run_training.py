"""
Run training module for cloud environment,
when ENV_MODE from .env is set to staging or prod.

References:
https://codelabs.developers.google.com/vertex-p2p-training#5
"""

from dotenv import dotenv_values
from google.cloud import aiplatform
from google.api_core.exceptions import NotFound

ENVS = dotenv_values()


def main():

  my_job = aiplatform.CustomContainerTrainingJob(
      display_name=ENVS["TRAINING_NAME"],
      container_uri=ENVS["TRAINING_IMAGE_URI"],
      staging_bucket=ENVS["BUCKET_NAME"],
      location=ENVS["REGION"],
      model_serving_container_image_uri=ENVS["INFERENCE_IMAGE_URI"],
      model_serving_container_ports=[int(ENVS["PORT"])],
      model_serving_container_predict_route=ENVS["PREDICT_ROUTE"],
      model_serving_container_health_route=ENVS["HEALTH_ROUTE"],
      model_serving_container_environment_variables={
          "ENV_MODE": ENVS["ENV_MODE"],
          "MODEL_VERSION_ALIAS": ENVS["MODEL_VERSION_ALIAS"],
          "PREPROCESSORS_PATH": ENVS["PREPROCESSORS_PATH"],
          "BATCH_SIZE": ENVS["BATCH_SIZE"],
          "PORT": ENVS["PORT"],
          "PREDICT_ROUTE": ENVS["PREDICT_ROUTE"],
          "EVAL_ROUTE": ENVS["EVAL_ROUTE"],
          "HEALTH_ROUTE": ENVS["HEALTH_ROUTE"],
      },
  )

  model_id = ENVS["MODEL_ID"]
  parent_model = None
  try:
    model = aiplatform.Model(ENVS["MODEL_ID"])
  except NotFound:
    model_id, parent_model = parent_model, model_id

  model = my_job.run(
      model_id=model_id,
      # model_display_name=ENVS["MODEL_ID"],
      model_version_aliases=[ENVS["MODEL_VERSION_ALIAS"]],
      parent_model=parent_model,
      replica_count=1,
      machine_type=ENVS[f'{ENVS["ENV_MODE"].upper()}_TRAINING_MACHINE_TYPE'],
      # accelerator_type='NVIDIA_TESLA_T4',
      # accelerator_count=1,
      base_output_dir=ENVS["BASE_OUTPUT_DIR"],
      enable_dashboard_access=True,
      environment_variables=ENVS,
  )

  return model


if __name__ == "__main__":
  main()
