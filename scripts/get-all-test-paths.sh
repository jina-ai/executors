#!/bin/bash
# find all the examples with changed code
# run the tests in that directory
set -ex

changed_folders=()
root=`pwd`

for changed_file in $CHANGED_FILES; do
#  echo changed $changed_file

  file_base_dir=$(dirname $changed_file)
#  echo checking $file_base_dir
  cd $file_base_dir

  # only if the folder has a tests or a Dockerfile but excluding integration tests (always run & separate)
  if [[ -f "Dockerfile" || -d "tests/" ]]; then
    if [[ ! " ${changed_folders[@]} " =~ " ${file_base_dir} " ]]; then
#      echo adding $file_base_dir
      if [[ $file_base_dir != "." ]]; then
        changed_folders+=(${file_base_dir})
      fi
    fi
  fi

  cd $root
done

#echo will run tests on ${changed_folders[@]}
printf '%s\n' "${changed_folders[@]}" | jq -R . | jq -cs .