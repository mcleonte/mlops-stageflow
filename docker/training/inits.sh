#!/bin/bash

if [ $ENV_MODE = "debug" ] || [ $ENV_MODE = "dev" ] ; then

    source /.env

elif [ $ENV_MODE = "staging" ] || [ $ENV_MODE = "prod" ] ; then

    mkdir /gcs && mkdir /gcs/$BUCKET_NAME

    gcsfuse \
        --implicit-dirs \
        $BUCKET_NAME \
        /gcs/$BUCKET_NAME

fi
