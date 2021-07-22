FROM jinaai/jina:2.0.8-standard
RUN apt-get update && apt-get install -y git

COPY . /big_transfer/
WORKDIR /big_transfer

RUN pip install .

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]