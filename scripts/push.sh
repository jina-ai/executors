#!/bin/bash
apt-get update && apt-get install -y jq

push_dir=$1
echo pushing $push_dir
cd $push_dir

exec_name=${PWD##*/}
exec_uuid=`echo $uuids | jq -r .$exec_name`
echo "::add-mask::$exec_uuid"
echo UUID=`head -c 3 <(echo $exec_uuid)`

exec_secret=`echo $secrets | jq -r .$exec_name`
echo "::add-mask::$exec_secret"
echo SECRET=`head -c 3 <(echo $exec_secret)`

jina hub push --force $exec_uuid --secret $exec_secret .