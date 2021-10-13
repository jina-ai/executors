FROM jinaai/jina:2-py37-perf

# install git
RUN apt-get -y update && apt-get install -y wget ffmpeg libsndfile-dev

# install requirements before copying the workspace
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# setup the workspace
COPY . /workdir
WORKDIR /workdir

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
