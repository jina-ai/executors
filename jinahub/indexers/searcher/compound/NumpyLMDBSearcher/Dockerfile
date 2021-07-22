FROM jinaai/jina:2.0 as base
# install the third-party requirements
RUN apt-get update && apt-get install --no-install-recommends -y gcc build-essential git

# setup the workspace
COPY . /workspace
WORKDIR /workspace

RUN pip install -r requirements.txt

FROM base as test
# no tests here. Check integration tests
RUN echo no tests here. Check integration tests

FROM base as entrypoint
ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
