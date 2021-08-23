FROM jinaai/jina:2.0.18 as base
# install and upgrade pip

# setup the workspace
COPY . /workspace
WORKDIR /workspace

FROM base as entrypoint
ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
