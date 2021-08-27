#!/bin/bash
# find all the examples with changed code
# run the tests in that directory
set -ex

containers=()
gpu_folders=()
root=`pwd`

for changed_file in $CHANGED_FILES; do

  file_base_dir=$(dirname $changed_file)
  if [ $(basename $file_base_dir) = "tests" ]; then
    file_base_dir=$(dirname "$file_base_dir")
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
echo "::set-output name=matrix::$(printf '%s\n' "${gpu_folders[@]}" | jq -R . | jq -cs .)"