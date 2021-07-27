FROM jinaai/jina:2.0.10 as base

# install git
RUN apt-get -y update && apt-get -y install redis-server git

# install requirements before copying the workspace
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt

# setup the workspace
COPY . /workspace

WORKDIR /workspace

FROM base as entrypoint
ENTRYPOINT ["/bin/bash", "/workspace/scripts/entrypoint.sh"]
