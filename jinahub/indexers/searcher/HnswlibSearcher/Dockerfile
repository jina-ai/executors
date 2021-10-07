# DO NOT DELETE THIS DOCKERFILE
# required because hnsw requires a specific gcc

FROM jinaai/jina:2-py37-perf

COPY . /workspace
WORKDIR /workspace

# install GCC compiler
RUN apt-get update && apt-get install --no-install-recommends -y build-essential \
    && rm -rf /var/lib/apt/lists/*

# install the third-party requirements
RUN pip install --compile --no-cache-dir \
     -r requirements.txt

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
