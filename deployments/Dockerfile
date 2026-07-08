FROM vllm/vllm-openai:v0.9.1

RUN pip install ms-swift@git+https://github.com/modelscope/ms-swift.git@v3.5.3
RUN pip install qwen-vl-utils==0.0.11
RUN pip install decord==0.6.0
RUN pip install google-cloud-storage
RUN pip install --upgrade "transformers==4.51.0"

RUN pip install "starlette<1.0.0,>=0.30.0" "fastapi<1.0.0"

RUN python3 -c "import os, prometheus_fastapi_instrumentator as p; path = os.path.join(os.path.dirname(p.__file__), 'routing.py'); data = open(path).read(); open(path, 'w').write(data.replace('for route in routes:', 'for route in routes:\n        if not hasattr(route, \"path\"): continue'))"

RUN apt-get update && apt-get install -y curl gnupg apt-transport-https ca-certificates && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor --yes -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update && apt-get install -y google-cloud-cli && \
    rm -rf /var/lib/apt/lists/* && apt-get clean

RUN gsutil --version

RUN mkdir -p /opt/ml/model
RUN chmod -R 755 /opt/ml

# Model is downloaded from GCS in a separate Cloud Build step (which has
# credentials) into model_data/ in the build context, then copied here.
# Do not use RUN gsutil here: RUN steps have no Cloud Build credentials and
# will fail with "Anonymous caller does not have storage.objects.list access".
COPY model_data/ /opt/ml/model/
RUN echo "====> Baked model contents:" && ls -lh /opt/ml/model

ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g $GROUP_ID vllmuser && \
    useradd -m -u $USER_ID -g $GROUP_ID vllmuser

ENV PORT=8080
EXPOSE $PORT

WORKDIR /home/vllmuser/app

RUN chown -R vllmuser:vllmuser /opt/ml

COPY entrypoint.sh /home/vllmuser/app/
RUN chmod +x /home/vllmuser/app/entrypoint.sh && \
    chown -R vllmuser:vllmuser /home/vllmuser

USER vllmuser

ENTRYPOINT ["/home/vllmuser/app/entrypoint.sh"]
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD curl -f http://localhost:$PORT/health || exit 1