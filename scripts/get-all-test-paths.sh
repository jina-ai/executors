#!/bin/bash
# find all the examples with changed code
# run the tests in that directory
set -ex

changed_folders=()

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

  # only if the folder has a tests or a Dockerfile but excluding integration tests (always run & separate)
  if [[ -f "${file_base_dir}/Dockerfile" || -d "${file_base_dir}/tests/" ]]; then
    if [[ ! " ${changed_folders[@]} " =~ " ${file_base_dir} " ]]; then
      if [[ $file_base_dir != "." ]]; then
        changed_folders+=(${file_base_dir})
      fi
    fi
  fi
done

output=$(jq --compact-output --null-input '$ARGS.positional' --args "${changed_folders[@]}")
echo "::set-output name=matrix::${output}"