#!/bin/bash

source .env


gcsfuse \
    $BUCKET_NAME \
    ./gcs/$BUCKET_NAME/
