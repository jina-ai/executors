FROM jinaai/jina:2.0.20-perf

RUN apt-get -y update && apt-get install -y git

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./transformer-text-encoder/
WORKDIR ./transformer-text-encoder

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
