FROM jinaai/jina:2.0.20-standard
RUN apt-get update && apt-get install -y git

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /big_transfer/
WORKDIR /big_transfer

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]