#!/usr/bin/bash

#print user info
echo "$(id)"

# Define mlflow
export MLFLOW_TRACKING_URI=file:/project/mlruns
echo ${MLFLOW_TRACKING_URI}

# Define home for runai
export USER="${USER:=`whoami`}"
export HOME=/home/$USER

export PHYSIONET_SLEEP_PATH=/data/
# parse arguments
CMD=""
for i in $@; do
  if [[ $i == *"="* ]]; then
    ARG=${i//=/ }
    CMD=$CMD"--$ARG "
  else
    CMD=$CMD"$i "
  fi
done

# execute comand
echo $CMD
$CMD
