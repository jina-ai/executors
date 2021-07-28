FROM jinaai/jina:master-py37-perf
RUN apt-get update && apt-get install -y python3.7 python3.7-dev python3-pip git

COPY . ./image_normalizer/
WORKDIR ./image_normalizer

RUN pip install -r requirements.txt

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
