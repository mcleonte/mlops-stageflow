#!/bin/bash

source .env

if [ $1 = "training" ] ; then

    if [ $ENV_MODE = "debug" ] || [ $ENV_MODE = "dev" ] ; then

        # run model evaluation locally
        docker run \
            --rm \
            --privileged \
            -v $(pwd)/docker/$1/src/:/src/ \
            -v $(pwd)/.env:/.env \
            -v $(pwd)/gcs/:/gcs/ \
            -e ENV_MODE=$ENV_MODE \
            -e KAGGLE_KEY=$KAGGLE_KEY \
            $TRAINING_IMAGE_URI \
            eval

    elif [ $ENV_MODE = "staging" ] || [ $ENV_MODE = "prod" ] ; then

        # do model evaluation with
        # a batch prediction on Vertex AI
        python src/test_training.py

    fi

elif [ $1 = "inference" ] ; then

    if [ $ENV_MODE = "debug" ] || [ $ENV_MODE = "dev" ] ; then

        curl \
            -X POST \
            -H "Content-Type: application/json" \
            -d '{"instances":["when must i pay?","card not working"]}' \
            http://localhost:$PORT$PREDICT_ROUTE


    elif [ $ENV_MODE = "staging" ] || [ $ENV_MODE = "prod" ] ; then

        # do online prediction
        # on Vertex AI endpoint
        # with test requests
        python src/test_inference.py

    fi

fi
