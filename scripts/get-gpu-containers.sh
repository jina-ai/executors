#!/bin/bash
# find all the examples with changed code
# run the tests in that directory
set -ex

containers=()
changed_folders=()
root=`pwd`

for changed_file in $CHANGED_FILES; do

  file_base_dir=$(dirname $changed_file)
  if [ $(basename $file_base_dir) = "tests" ]; then
    file_base_dir=$(dirname "$file_base_dir")
  fi
  cd $file_base_dir

  if [[ ! " ${changed_folders[@]} " =~ " ${file_base_dir} " ]]; then
    if [[ $file_base_dir != "." ]]; then
      x=( $(find . -type f -iname "Dockerfile.cuda*"))
      for item in "${x[@]}"; {
        tag="${item##*.}"
        echo tag $tag
        echo "GPU executor found in " $changed_folder
        echo "Building docker image." $item
        name=$(echo "$(basename $file_base_dir)" | tr '[:upper:]' '[:lower:]')
        docker build -t "localhost:5000/$name:$tag" -f $item .
        docker push "localhost:5000/$name:$tag"
        containers+=("localhost:5000/$name:$tag")
      }
      changed_folders+=(${file_base_dir})
    fi
  fi
  cd $root
done

#echo will run tests on ${changed_folders[@]}
echo "::set-output name=matrix::$(printf '%s\n' "${containers[@]}" | jq -R . | jq -cs .)"