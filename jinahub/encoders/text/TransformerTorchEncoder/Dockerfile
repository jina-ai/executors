FROM jinaai/jina:master-perf

RUN apt-get -y update && apt-get install -y git \
    && rm -rf /var/lib/apt/lists/*

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# install requirements before copying the workspace
RUN pip install -r requirements.txt
RUN pip install -r requirements.txt


ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
