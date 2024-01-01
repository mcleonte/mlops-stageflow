### Description
This Machine Learning project simplifies the transition between different
development stages. It features an intuitive mechanism for toggling between
environments, ensuring a streamlined workflow for both local and cloud-based
development on Google Cloud Platform.

### Key Features
- Easy Environment Switching: Toggle `ENV_MODE` in the `.env` file to seamlessly
switch between debug, dev, staging, and prod environments.
- centralized configuration: `.env` centralizes critical variables for automated
  resource naming, model configuration and versioning, and differentiated
  infrastructure specifications.
- simplified command structure: execute build, run, and test commands with ease
  across training and inference stages.

### How to Use
- clone the repository
- configure the `.env` file according to your environment needs
- use the following commands to manage your project:

  training:
  - `./scripts/build.sh training`
  - `./scripts/run.sh training`
  - `./scripts/test.sh training`

  inference:
  - `./scripts/build.sh inference`
  - `./scripts/run.sh inference`
  - `./scripts/test.sh inference`

### Upcoming Features
  - preprocessors versioning
  - training evaluation for staging and production stages using Vertex AI
    evaluation job (programatic implementation currently not supported)
  - improved logs for debugging stage
  - support for diferentiated training dataset for production stage
