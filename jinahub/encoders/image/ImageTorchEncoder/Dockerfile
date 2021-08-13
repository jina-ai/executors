FROM jinaai/jina:2.0.3

RUN apt-get update && apt install -y git

COPY . ./image_torch_encoder/
WORKDIR ./image_torch_encoder

RUN pip install -r requirements.txt

# setup the workspace
COPY . /workspace
WORKDIR /workspace

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
