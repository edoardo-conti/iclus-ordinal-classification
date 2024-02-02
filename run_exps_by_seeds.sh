#!/bin/bash

# Specifica il range di seed che vuoi utilizzare
for seed in {1..100}; do
    # Costruisci il comando con il seed corrente
    command="python experiments_launcher.py --exps_json exps/experiments.json --dataset dataset_tfrecords/ --no_gridsearch --workers 7 --verbose 1 --seed $seed"
    
    # Stampa il comando (opzionale)
    echo "Running with seed: $seed"
    echo "Command: $command"
    
    # Esegui il comando
    $command
done