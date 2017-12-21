#!/bin/bash

for para in 100 1000 2000 4000 6000 8000 10000 15000 20000 25000
do
    command="python lstm-imdb_aff.py --model sl --source imdb --train_size $para"
    echo $command
    $command
done

