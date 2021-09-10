FROM jinaai/jina:2-py37-perf
RUN apt-get update && apt-get install -y git

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /workdir/
WORKDIR /workdir

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]