#!/bin/bash
# find all the examples with changed code
# run the tests in that directory
set -ex

changed_folders=()
root=`pwd`

for changed_file in $CHANGED_FILES; do
# echo changed $changed_file

  file_base_dir=$(dirname $changed_file)
  if [ $(basename $file_base_dir) = "tests" ]; then
    file_base_dir=$(dirname "$file_base_dir")
  fi
#  echo checking $file_base_dir
  cd $file_base_dir

  # only if the folder has a tests or a Dockerfile but excluding integration tests (always run & separate)
  if [[ -f "${file_base_dir}/Dockerfile" || -d "${file_base_dir}/tests/" ]]; then
    if [[ ! " ${changed_folders[@]} " =~ " ${file_base_dir} " ]]; then
      if [[ $file_base_dir != "." ]]; then
        changed_folders+=(${file_base_dir})
      fi
    fi
  fi

  cd $root
done

output=$(jq --compact-output --null-input '$ARGS.positional' --args "${changed_folders[@]}")
echo "::set-output name=matrix::${output}"