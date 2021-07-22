#!/bin/bash
# find all the examples with changed code
# run the tests in that directory
set -ex

test_dir=$1
echo testing $test_dir
cd $test_dir

if [[ -d "tests/" ]]; then
  echo running tests in $test_dir
  python -m venv .venv
  source .venv/bin/activate
  pip install pytest pytest-mock
  pip install -r requirements.txt
  pytest -s -v tests/
  local_exit_code=$?
  deactivate
else
  echo no tests or Dockerfile in $test_dir
  local_exit_code=0
fi

if [[ ! $local_exit_code == 0 ]]; then
  EXIT_CODE=$local_exit_code
  echo $test_dir failed. local_exit_code = $local_exit_code, exit = $EXIT_CODE
fi

echo final exit code = $EXIT_CODE
exit $EXIT_CODE
