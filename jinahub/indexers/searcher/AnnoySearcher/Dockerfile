FROM jinaai/jina:2-py37-perf as base

# install and upgrade pip
RUN apt-get update && apt-get install -y git g++

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# install Jina and third-party requirements
RUN python3.7 -m pip install -r requirements.txt

FROM base as entrypoint
ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
