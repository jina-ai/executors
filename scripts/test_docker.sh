#!/bin/bash
# find all the examples with changed code
# run the docker test in that directory
set -ex

test_dir=$1
echo testing $test_dir
cd $test_dir

# assume failure
local_exit_code=1

# test docker image actually runs
if [[ -f "Dockerfile" ]]; then
  python -m venv .venv
  source .venv/bin/activate
  pip install wheel docker jina
  pip install -r requirements.txt

  if [[ -f "tests/requirements.txt" ]]; then
    pip install -r tests/requirements.txt
  fi

  docker build -t foo .
  if [[ -f "tests/pre-docker.sh" ]]; then # allow entrypoint for any pre-docker run operations, liek downloading a model to mount
    bash tests/pre-docker.sh
  fi
  if [[ -f "tests/docker_args.txt" ]]; then # allow args to be passed to the `jina pea`
      ARGS=`cat tests/docker_args.txt`
    else
      ARGS=""
  fi
  nohup jina pea --uses docker://foo:latest $ARGS > nohup.out 2>&1 &
  PID=$!
  sleep 10
  if ps -p $PID > /dev/null;
  then
    kill -9 $PID
    docker rm -f $(docker ps -a -q)
    docker rmi foo:latest
    local_exit_code=0
  else
    echo "jina pea --uses docker://foo:latest" could NOT start
  fi
  echo ~~~~~~~OUTPUT BELOW~~~~~~~
  cat nohup.out
else
  echo no Dockerfile, nothing to test
  local_exit_code=0
fi

echo final exit code = $local_exit_code
exit $local_exit_code