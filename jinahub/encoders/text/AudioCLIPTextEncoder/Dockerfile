FROM jinaai/jina:2.0.10-perf

# install git
RUN apt-get -y update && apt-get install -y git \
    && rm -rf /var/lib/apt/lists/*

# install requirements before copying the workspace
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

# setup the workspace
COPY . /workspace
WORKDIR /workspace

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
