FROM jinaai/jina:2-py37-perf as base

# install the third-party requirements
RUN apt-get update && apt-get install --no-install-recommends -y gcc build-essential git

# setup the workspace
COPY . /workspace
WORKDIR /workspace

RUN pip install -r requirements.txt
RUN cp -r /usr/local/lib/python3.7/site-packages/jinahub/ /usr/local/lib/python3.7/site-packages/jina_executors # required until we fix import issue in core


FROM base as test
# no tests here. Check integration tests
RUN echo no tests here. Check integration tests

FROM base as entrypoint
ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
