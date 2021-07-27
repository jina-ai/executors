FROM jinaai/jina:2.0.10

# install git
RUN apt-get -y update && apt-get install -y git gcc ffmpeg libsm6 libxext6 libgl1-mesa-glx

# install requirements before copying the workspace
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt

# setup the workspace
COPY . /workspace
WORKDIR /workspace

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
