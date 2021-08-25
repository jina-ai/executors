FROM jinaai/jina:2.0.20-standard
RUN apt-get update && apt-get install -y git

COPY . /big_transfer/
WORKDIR /big_transfer

RUN pip install -r requirements.txt

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]