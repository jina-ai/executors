FROM jinaai/jina:master-py37-perf
# install the third-party requirements
RUN apt-get update && apt-get install -y git

# setup the workspace
COPY . /workspace
WORKDIR /workspace

RUN python3.7 -m pip install -r requirements.txt

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
