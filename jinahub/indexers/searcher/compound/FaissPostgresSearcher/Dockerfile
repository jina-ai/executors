FROM jinaai/jina:master-py37-perf as base

# install and upgrade pip
RUN apt-get update && apt-get install -y git
RUN python3.7 -m pip install --upgrade pip

# setup the workspace
COPY requirements.txt /requirements.txt

# install Jina and third-party requirements
RUN python3.7 -m pip install -r requirements.txt
RUN cp -r /usr/local/lib/python3.7/site-packages/jinahub/ /usr/local/lib/python3.7/site-packages/jina_executors # required until we fix import issue in core

COPY . /workspace
WORKDIR /workspace

FROM base as entrypoint
EXPOSE 5432
ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
