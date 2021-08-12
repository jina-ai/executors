#!/bin/bash
# find all the examples with changed code
# run the tests in that directory
set -ex

test_dir=$1
echo testing $test_dir
cd $test_dir

# assume failure
local_exit_code=1

# test docker image actually runs
if [[ -f "Dockerfile" ]]; then
  docker build -t foo .
  pip install docker
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
    docker stop $(docker ps -a -q)
    local_exit_code=0
  else
    echo "jina pea --uses docker://foo:latest" could NOT start
  fi
  echo ~~~~~~~OUTPUT BELOW~~~~~~~
  cat nohup.out

fi

echo final exit code = $EXIT_CODE
exit $local_exit_code