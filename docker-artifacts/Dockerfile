FROM vllm/vllm-openai:v0.9.1

# Install system dependencies FIRST (including google-cloud-sdk)
RUN apt-get update && \
    apt-get install -y curl gnupg lsb-release && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update && \
    apt-get install -y google-cloud-sdk && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Install Python dependencies
RUN pip install ms-swift@git+https://github.com/modelscope/ms-swift.git@v3.5.3
RUN pip install qwen-vl-utils==0.0.11
RUN pip install decord==0.6.0
RUN pip install google-cloud-storage
RUN pip install gsutil
RUN pip install --upgrade git+https://github.com/huggingface/transformers.git@v4.52.4
# RUN pip install "vllm>=0.7.2" -U

# # Make server compatible with SageMaker Hosting contract
# RUN sed -i 's|/v1/models|/ping|g' /usr/local/lib/python3.12/dist-packages/swift/llm/infer/deploy.py
# RUN sed -i 's|/v1/chat/completions|/invocations|g' /usr/local/lib/python3.12/dist-packages/swift/llm/infer/deploy.py
# RUN sed -i 's|/v1/completions|/invocations/completions|g' /usr/local/lib/python3.12/dist-packages/swift/llm/infer/deploy.py

# Create SageMaker directory structure
RUN mkdir -p /opt/ml/model
RUN chmod -R 755 /opt/ml

# Create a non-root user with specific UID/GID (1000 is common for first user)
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g $GROUP_ID vllmuser && \
    useradd -m -u $USER_ID -g $GROUP_ID vllmuser

ENV PORT=8080
EXPOSE $PORT

# Create working directory for the application
WORKDIR /home/vllmuser/app

# script runs as a non-root user, ensure ownership
RUN chown -R vllmuser:vllmuser /opt/ml

# Copy entrypoint script and set permissions
COPY vertex-entrypoint.sh /home/vllmuser/app/
RUN chmod +x /home/vllmuser/app/vertex-entrypoint.sh && \
    chown -R vllmuser:vllmuser /home/vllmuser

# Switch to non-root user
USER vllmuser

ENTRYPOINT ["/home/vllmuser/app/vertex-entrypoint.sh"]
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD curl -f http://localhost:$PORT/health || exit 1