#!/bin/bash

source .env

if [ $1 = "training" ] ; then

    IMAGE_URI=$TRAINING_IMAGE_URI

elif [ $1 = "inference" ] ; then

    IMAGE_URI=$INFERENCE_IMAGE_URI

fi


docker build \
    -t $IMAGE_URI \
    --build-arg MODEL_VERSION_ALIAS=$MODEL_VERSION_ALIAS \
    --build-arg BUCKET_NAME=$BUCKET_NAME \
    -f docker/$1/Dockerfile .


if [ $ENV_MODE = "staging" ] ; then

    echo "Pushing to GCP Artifact Registry..."

    # ensure Artifact Registry repository

    if ! gcloud artifacts repositories describe $REPO_NAME --location=$REGION ; then

        echo "Repository '"$REPO_NAME"' does not exist in "$REGION". Creating new repository..."

        gcloud artifacts repositories create \
            $REPO_NAME \
            --repository-format=docker \
            --location=$REGION \
            --description="Docker repository"
    fi

    # configure Docker

    gcloud auth configure-docker \
        $REGION-docker.pkg.dev

    docker push $IMAGE_URI

fi
