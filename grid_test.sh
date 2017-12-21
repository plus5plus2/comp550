#!/bin/bash














for para in 1 5 10 15 25
do
    command="python lstm-imdb_aff.py --model cl --source imdb --epoch $para"
    echo $command
    $command
done

for para in 500 400 300 200 100 50 30
do
    command="python lstm-imdb_aff.py --model cl --source imdb --batch $para"
    echo $command
    $command
done

for para in 1 2 3 4 5
do
    command="python lstm-imdb_aff.py --model cl --source imdb --layers $para"
    echo $command
    $command
done

for para in 2 4 8 16 32 64 128 256 512 1024
do
    command="python lstm-imdb_aff.py --model cl --source imdb --neurals $para"
    echo $command
    $command
done

for para in 100 1000 2000 4000 6000 8000 10000 15000 20000 25000
do
    command="python lstm-imdb_aff.py --model cl --source imdb --train_size $para"
    echo $command
    $command
done

for para in 1 5 10 15 25
do
    command="python lstm-imdb_aff.py --model sl --source imdb --epoch $para"
    echo $command
    $command
done

for para in 500 400 300 200 100 50 30
do
    command="python lstm-imdb_aff.py --model sl --source imdb --batch $para"
    echo $command
    $command
done

for para in 1 2 3 4 5
do
    command="python lstm-imdb_aff.py --model sl --source imdb --layers $para"
    echo $command
    $command
done

for para in 2 4 8 16 32 64 128 256 512 1024
do
    command="python lstm-imdb_aff.py --model sl --source imdb --neurals $para"
    echo $command
    $command
done

for para in 100 1000 2000 4000 6000 8000 10000 15000 20000 25000
do
    command="python lstm-imdb_aff.py --model sl --source imdb --train_size $para"
    echo $command
    $command
done




exit


for para in 1 5 10 15 25
do
    command="python lstm-imdb_aff.py --model cl --source afd --epoch $para"
    echo $command
    $command
done

for para in 500 400 300 200 100 50 30
do
    command="python lstm-imdb_aff.py --model cl --source afd --batch $para"
    echo $command
    $command
done

for para in 1 2 3 4 5
do
    command="python lstm-imdb_aff.py --model cl --source afd --layers $para"
    echo $command
    $command
done

for para in 2 4 8 16 32 64 128 256 512 1024
do
    command="python lstm-imdb_aff.py --model cl --source afd --neurals $para"
    echo $command
    $command
done

for para in 100 1000 2000 4000 6000 8000 10000 15000 20000 25000
do
    command="python lstm-imdb_aff.py --model cl --source afd --train_size $para"
    echo $command
    $command
done

for para in 1 5 10 15 25
do
    command="python lstm-imdb_aff.py --model sl --source afd --epoch $para"
    echo $command
    $command
done

for para in 500 400 300 200 100 50 30
do
    command="python lstm-imdb_aff.py --model sl --source afd --batch $para"
    echo $command
    $command
done

for para in 1 2 3 4 5
do
    command="python lstm-imdb_aff.py --model sl --source afd --layers $para"
    echo $command
    $command
done

for para in 2 4 8 16 32 64 128 256 512 1024
do
    command="python lstm-imdb_aff.py --model sl --source afd --neurals $para"
    echo $command
    $command
done

for para in 100 1000 2000 4000 6000 8000 10000 15000 20000 25000
do
    command="python lstm-imdb_aff.py --model sl --source afd --train_size $para"
    echo $command
    $command
done



