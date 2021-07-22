FROM jinaai/jina:2.0.7

# install git
RUN apt-get -y update && apt-get install -y git wget && apt-get install -y libsndfile-dev

# install requirements before copying the workspace
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt

# setup the workspace
COPY . /workspace
WORKDIR /workspace


RUN ./scripts/download_model.sh

ENV PYTHONPATH=/workspace
ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
