#!/bin/bash
# find all the examples with changed code
# run the tests in that directory
set -ex

gpu_folders=()
root=`pwd`

for changed_file in $CHANGED_FILES; do

  file_base_dir=$(dirname $changed_file)
  # Test folder changes
  if [ $(basename $file_base_dir) = "tests" ]; then
    file_base_dir=$(dirname "$file_base_dir")
  fi
  # Changes in subfolder of test folder (e.g. unit_test/integration)
  if [ $(basename $(dirname "$file_base_dir")) = "tests" ]; then
    file_base_dir=$(dirname $(dirname "$file_base_dir"))
  fi
  cd $file_base_dir

  if [[ ! " ${gpu_folders[@]} " =~ " ${file_base_dir} " ]]; then
    if [[ $file_base_dir != "." ]]; then
      if [[ -f "Dockerfile.gpu" ]]; then
        echo "GPU executor found in " $file_base_dir
        gpu_folders+=(${file_base_dir})
      fi
    fi
  fi
  cd $root
done

#echo will store gpu_folders in output matrix
output=$(jq --compact-output --null-input '$ARGS.positional' --args "${gpu_folders[@]}")
echo "::set-output name=matrix::${output}"
