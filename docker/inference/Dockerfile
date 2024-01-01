FROM gcr.io/deeplearning-platform-release/tf2-gpu.py310:latest

ARG BUCKET_NAME

ARG MODEL_VERSION_ALIAS

WORKDIR /app

COPY docker/inference/requirements.txt .

RUN /opt/conda/bin/pip install -r requirements.txt

COPY docker/inference/app.py .

COPY gcs/gcp-mlops-email-classification/preprocessors/ preprocessors/

COPY gcs/$BUCKET_NAME/models/$MODEL_VERSION_ALIAS/model/ models/$MODEL_VERSION_ALIAS/model/

ENTRYPOINT ["python"]

CMD ["app.py"]
