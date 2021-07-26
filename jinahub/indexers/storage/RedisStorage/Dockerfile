FROM jinaai/jina:2.0.10

# install git
RUN apt-get -y update && apt-get -y install redis-server

# install requirements before copying the workspace
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt

# setup the workspace
COPY . /workspace

WORKDIR /workspace

# for testing the image
RUN redis-server --bind 0.0.0.0 --port 6379:6379 --daemonize yes && pip install pytest && pip install pytest-mock && pytest

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
