#!/bin/sh
# Use the paper's N/K/deadline ranges with explicitly NEW seeded datasets.
# TRIALS defaults to the paper's 10000 random assignments per baseline run.
set -eu
bin=${1:-bin/mcc_scheduler}
trials=${TRIALS:-10000}
seed=${SEED:-1}
for cores in 3 6; do
    if [ "$cores" -eq 3 ]; then
        deadlines='100 150 170 210 250 330 400 450 500 550'
    else
        deadlines='60 80 110 140 180 240 300 350 380 420'
    fi
    tasks=11
    for deadline in $deadlines; do
        code=0
        "$bin" --experiment --tasks "$tasks" --cores "$cores" \
            --deadline "$deadline" --seed "$seed" --trials "$trials" || code=$?
        # A method missing its deadline is a reported experimental outcome,
        # not a script failure. Invalid input/runtime errors must stop the sweep.
        if [ "$code" -gt 1 ]; then exit "$code"; fi
        tasks=$((tasks + 10))
    done
done
