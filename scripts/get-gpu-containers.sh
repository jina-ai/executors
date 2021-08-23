#!/bin/bash
# find all the examples with changed code
# run the tests in that directory
set -ex

containers=()
root=`pwd`

for changed_folder in $CHANGED_FOLDERS; do

  cd changed_folder

  x=( $(find . -type f -iname "Dockerfile.cuda*"))
  for item in "${x[@]}"; {
    tag="${item##*.}"
    echo "GPU executor found in " $changed_folder
    echo "Building docker image." $item
    docker build -t $(basename $changed_folder):$tag -f $item .
    containers+=(${$(basename $changed_folder):$tag})
  }

  cd $root
done

#echo will run tests on ${changed_folders[@]}
printf '%s\n' "${containers[@]}" | jq -R . | jq -cs .