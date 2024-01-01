FROM gcr.io/deeplearning-platform-release/tf2-gpu.py310:latest

WORKDIR /

COPY docker/training/requirements.txt .

RUN /opt/conda/bin/pip install -r requirements.txt

COPY docker/training/inits.sh entrypoint.sh ./

RUN chmod +x inits.sh && chmod +x entrypoint.sh

COPY docker/training/src/ ./src/

ENTRYPOINT ["./entrypoint.sh"]

CMD ["train"]
