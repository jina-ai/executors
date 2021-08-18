FROM jinaai/jina:master-py37-perf
RUN apt-get update && apt-get install -y git

COPY . ./image_normalizer/
WORKDIR ./image_normalizer

RUN pip install -r requirements.txt

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
