#!/bin/bash

results_dir="./results"
logs_dir="./logs"
backup_dir="./backup"
weights_dir="./weights"
split_data_file="./splitdata.pkl"
checkpoint_file="./run_checkpoint.txt"

delete_directory() {
    local dir="$1"
    if [ -d "$dir" ]; then
        rm -r "$dir"
        echo "Directory '$dir' deleted"
    else
        echo "Directory $dir do not exists"
    fi
}

delete_file() {
    local file="$1"
    if [ -f "$file" ]; then
        rm "$file"
        echo "File $file deleted"
    else
        echo "File $file do not exists"
    fi
}

# delete directories
delete_directory "$results_dir"
delete_directory "$logs_dir"
delete_directory "$backup_dir"
delete_directory "$weights_dir"

# delete files
delete_file "$split_data_file"
delete_file "$checkpoint_file"