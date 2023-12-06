#!/bin/bash

results_dir="./results"
logs_dir="./logs"
backup_dir="./backup"

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

# delete files