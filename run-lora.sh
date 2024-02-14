#!/bin/bash

RANK=$SLURM_PROCID
WORLD_SIZE=$SLURM_NTASKS
NUM_CLIENTS=$((WORLD_SIZE - 1))
ARGS="$@"
if [ "$RANK" == "0" ]; then
    echo "Starting KD server"
    python3 server.py $ARGS --num_clients $NUM_CLIENTS --rank $RANK
else
    sleep 3
    python3 lora-client.py $ARGS --num_clients $NUM_CLIENTS --rank $RANK
fi
