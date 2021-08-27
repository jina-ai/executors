#!/bin/bash
apt-get update && apt-get install -y jq curl

push_dir=$1

# empty change is detected as home directory
if [ -z "$push_dir" ]
then
      echo "\$push_dir is empty"
      exit 0
fi

echo pushing $push_dir
cd $push_dir

exec_name=${PWD##*/}
echo executor name is $exec_name

version=`jina -vf`
echo jina version $version

# clone file with secrets
echo "::add-mask::$token"
curl -H "Authorization: token $token" -H 'Accept: application/vnd.github.v3.raw' -O https://api.github.com/repos/jina-ai/executors-secrets/contents/secrets.json

exec_uuid=`cat secrets.json | jq -r '.[] | select(.Alias=="'$exec_name'").UUID8'`
echo "::add-mask::$exec_uuid"
echo UUID=`head -c 3 <(echo $exec_uuid)`

exec_secret=`cat secrets.json | jq -r '.[] | select(.Alias=="'$exec_name'").Secret'`
echo "::add-mask::$exec_secret"
echo SECRET=`head -c 3 <(echo $exec_secret)`

rm secrets.json

jina hub push --force $exec_uuid --secret $exec_secret -t gpu -f Dockerfile.gpu .
