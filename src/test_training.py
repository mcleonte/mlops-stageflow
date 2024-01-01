"""
Test training module for cloud environment,
when ENV_MODE from .env is set to staging or prod.
"""

from datetime import datetime

from dotenv import dotenv_values
from google.cloud import aiplatform

ENVS = dotenv_values()


def main():

  aiplatform.init(
      project=ENVS["PROJECT_ID"],
      location=ENVS["REGION"],
  )

  # endpoint = ensure_endpoint(
  #     display_name=ENVS["ENDPOINT_NAME"],
  # )[0]

  # test_path = f'.{ENVS["DATA_PATH"]}/test.csv'
  # texts = pd.read_csv(test_path)["narrative"]

  # texts = {
  #     "narative": texts["narrative"].to_list(),
  #     "product": texts["product"].to_list(),
  # }

  today = datetime.today()
  gcs_destination_prefix = f'gs://{ENVS["BUCKET_NAME"]}/daily'
  gcs_source = f'{gcs_destination_prefix}/{today.strftime("%Y%m%d")}.csv'
  model_name = f'{ENVS["MODEL_ID"]}@{ENVS["MODEL_VERSION_ALIAS"]}'
  job_display_name = f'{model_name}-{today.strftime("%Y%m%d-%H%M%S")}'
  machine_type = ENVS[f'{ENVS["ENV_MODE"].upper()}_TRAINING_MACHINE_TYPE']

  print("Batch prediction in progress...")
  batch = aiplatform.BatchPredictionJob.create(
      job_display_name=job_display_name,
      model_name=model_name,
      instances_format="csv",
      predictions_format="jsonl",
      gcs_source=gcs_source,
      gcs_destination_prefix=gcs_destination_prefix,
      # model_parameters: Dict | None = None,
      machine_type=machine_type,
      # accelerator_type: str | None = None,
      # accelerator_count: int | None = None,
      # starting_replica_count: int | None = None,
      # max_replica_count: int | None = None,
      generate_explanation=True,
      # explanation_metadata: Any | None = None,
      # explanation_parameters: Any | None = None,
      # labels: Dict[str, str] | None = None,
      batch_size=128,
      # model_monitoring_objective_config: Any | None = None,
      # model_monitoring_alert_config: Any | None = None,
      # analysis_instance_schema_uri: str | None = None,
  )
  batch.wait()

  print("Daily evaluation in progress...")
  # print(json.dumps(test))

  print("Batch job done, check bucket.")


if __name__ == "__main__":
  main()
