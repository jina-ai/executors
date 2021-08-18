FROM jinaai/jina:2.0.18 as base
# install and upgrade pip
RUN apt-get update && apt-get install -y git

# setup the workspace
COPY . /workspace
WORKDIR /workspace

FROM base as entrypoint
ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
