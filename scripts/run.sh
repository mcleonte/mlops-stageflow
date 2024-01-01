#!/bin/bash

source .env

if [ $1 = "training" ] ; then

    if [ $ENV_MODE = "debug" ] ; then

        # run Docker image localy in interactive mode
        docker run \
            --rm \
            --privileged \
            -v $(pwd)/docker/$1/src/:/src/ \
            -v $(pwd)/.env:/.env \
            -v $(pwd)/gcs/:/gcs/ \
            -e ENV_MODE=$ENV_MODE \
            -it \
            --entrypoint bash \
            $TRAINING_IMAGE_URI

    elif [ $ENV_MODE = "dev" ] ; then

        # run model training locally
        docker run \
            --rm \
            --privileged \
            -v $(pwd)/docker/$1/src/:/src/ \
            -v $(pwd)/.env:/.env \
            -v $(pwd)/gcs/:/gcs/ \
            -e ENV_MODE=$ENV_MODE \
            -e KAGGLE_KEY=$KAGGLE_KEY \
            $TRAINING_IMAGE_URI \
            train

    elif [ $ENV_MODE = "staging" ] || [ $ENV_MODE = "prod" ] ; then

        # run custom training job
        # on Vertex AI
        python src/run_training.py

    fi

elif [ $1 = "inference" ] ; then

    if [ $ENV_MODE = "debug" ] ; then

        docker run \
            --entrypoint bash \
            -it \
            --rm \
            -v $(pwd)/docker/inference/app.py:/app/app.py \
            -e ENV_MODE=$ENV_MODE \
            -e BUCKET_NAME=$BUCKET_NAME \
            -e PREPROCESSORS_PATH=$PREPROCESSORS_PATH \
            -e MODEL_VERSION_ALIAS=$MODEL_VERSION_ALIAS \
            -e BATCH_SIZE=$BATCH_SIZE \
            -e PREDICT_ROUTE=$PREDICT_ROUTE \
            -e EVAL_ROUTE=$HEALTH_ROUTE \
            -e HEALTH_ROUTE=$HEALTH_ROUTE \
            -e PORT=$PORT \
            -p $PORT:$PORT \
            $INFERENCE_IMAGE_URI

    elif [ $ENV_MODE = "dev" ] ; then

        docker run \
            --rm \
            -v $(pwd)/docker/inference/app.py:/app/app.py \
            -e BUCKET_NAME=$BUCKET_NAME \
            -e ENV_MODE=$ENV_MODE \
            -e PREPROCESSORS_PATH=$PREPROCESSORS_PATH \
            -e MODEL_VERSION_ALIAS=$MODEL_VERSION_ALIAS \
            -e BATCH_SIZE=$BATCH_SIZE \
            -e PREDICT_ROUTE=$PREDICT_ROUTE \
            -e EVAL_ROUTE=$HEALTH_ROUTE \
            -e HEALTH_ROUTE=$HEALTH_ROUTE \
            -e PORT=$PORT \
            -p $PORT:$PORT \
            $INFERENCE_IMAGE_URI

    elif [ $ENV_MODE = "staging" ] || [ $ENV_MODE = "prod" ] ; then

        # deploy custom model
        # to Vertex AI endpoint
        python src/run_inference.py

    fi

fi
