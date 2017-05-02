#!/bin/bash
COUNTER=100
while [ $COUNTER -ge 1 ]; do
    sleep 2
    COUNTER=`nvidia-smi -i 1 --query-gpu=utilization.gpu --format=csv,noheader`
    echo $COUNTER
    COUNTER=`tr -cd 0-9 <<<"$COUNTER"`
    echo $COUNTER
    echo The gpu utilization is $COUNTER
    # let COUNTER=COUNTER+1
done
echo 'GPU is empty'
nvidia-smi --query-gpu=utilization.memory --format=csv,noheader

time python scripts/train_model.py
