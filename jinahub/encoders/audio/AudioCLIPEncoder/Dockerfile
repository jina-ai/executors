FROM jinaai/jina:2-py37-perf

# install git
RUN apt-get -y update && apt-get install -y git wget libsndfile-dev

# install requirements before copying the workspace
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# setup the workspace
COPY . /workdir
WORKDIR /workdir


RUN ./scripts/download_model.sh

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
