FROM jinaai/jina:2.0.22

RUN apt-get update && apt install -y git

COPY . ./timm_image_encoder/
WORKDIR ./timm_image_encoder

RUN pip install -r requirements.txt

# setup the workspace
COPY . /workspace
WORKDIR /workspace

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
