FROM jinaai/jina:2.0 as base
# install the third-party requirements
RUN apt-get update && apt-get install -y python3.7 python3.7-dev python3-pip git

# setup the workspace
COPY . /workspace
WORKDIR /workspace

RUN python3.7 -m pip install -r requirements.txt

FROM base as entrypoint
ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
