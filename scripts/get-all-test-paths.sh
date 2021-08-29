#!/bin/bash
# find all the examples with changed code
# run the tests in that directory
set -ex

changed_folders=()

for changed_file in $CHANGED_FILES; do
  file_base_dir=$(dirname $changed_file)

  # only if the folder has a tests or a Dockerfile but excluding integration tests (always run & separate)
  if [[ -f "${file_base_dir}/Dockerfile" || -d "${file_base_dir}/tests/" ]]; then
    if [[ ! " ${changed_folders[@]} " =~ " ${file_base_dir} " ]]; then
      if [[ $file_base_dir != "." ]]; then
        changed_folders+=(${file_base_dir})
      fi
    fi
  fi
done

if [ ${#changed_folders[@]} -eq 0 ]; then
    echo "::set-output name=matrix:: []"
else
    printf "::set-output name=matrix:: %s\n" "${changed_folders[@]}" | jq -R . | jq -cs .
fi
