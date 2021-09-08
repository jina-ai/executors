FROM jinaai/jina:2-py37-perf

# install git
RUN apt-get -y update && apt-get install -y git

# install requirements before copying the workspace
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0.tar.gz#egg=en_core_web_sm

# setup the workspace
COPY . /workdir
WORKDIR /workdir

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
