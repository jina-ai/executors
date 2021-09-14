FROM jinaai/jina:2-py37-perf


RUN apt-get -y update && apt-get install -y ffmpeg libsndfile1

# install requirements before copying the workspace
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

# setup the workspace
COPY . /workspace
WORKDIR /workspace

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
