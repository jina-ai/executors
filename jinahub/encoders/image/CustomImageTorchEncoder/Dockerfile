FROM jinaai/jina:2.0.20

# install git
RUN apt-get -y update && apt-get install -y git

# install requirements before copying the workspace
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# required in order to have a default model for starting the image in CI
RUN python tests/model/external_model.py

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
