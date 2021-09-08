FROM jinaai/jina:2-py37-perf

RUN apt update && apt install -y git
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /workdir/
WORKDIR /workdir

# required in order to have a default model for starting the image in CI
RUN python tests/model/external_model.py

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
