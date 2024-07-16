#!/bin/bash

# Directories to be deleted
directories=(
    "./results"
    "./logs"
)

# Files to be deleted
files=(
    "./kfold_labels.pkl"
)

# Function to delete a directory
delete_directory() {
    local dir="$1"
    if [ -d "$dir" ]; then
        rm -r "$dir"
        echo "Directory '$dir' deleted"
    else
        echo "Directory '$dir' does not exist"
    fi
}

# Function to delete a file
delete_file() {
    local file="$1"
    if [ -f "$file" ]; then
        rm "$file"
        echo "File '$file' deleted"
    else
        echo "File '$file' does not exist"
    fi
}

# Delete directories
for dir in "${directories[@]}"; do
    delete_directory "$dir"
done

# Delete files
for file in "${files[@]}"; do
    delete_file "$file"
done