#!/bin/bash

# Script to move multivariate directory files to the new structure
# By placing multivariate inside the tables directory

echo "Moving multivariate directories to new structure..."

# Define paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OLD_MULTIVARIATE_DIR="${PROJECT_ROOT}/output/multivariate"
NEW_MULTIVARIATE_DIR="${PROJECT_ROOT}/output/tables/multivariate"

# Check if old directory exists
if [ ! -d "$OLD_MULTIVARIATE_DIR" ]; then
    echo "Old multivariate directory not found at ${OLD_MULTIVARIATE_DIR}"
    echo "No migration needed."
    exit 0
fi

# Create new directory structure if it doesn't exist
mkdir -p "${NEW_MULTIVARIATE_DIR}"

# Check if new directory already has content
if [ "$(ls -A ${NEW_MULTIVARIATE_DIR})" ]; then
    echo "Warning: New multivariate directory already has content."
    read -p "Continue anyway? This may overwrite files. (y/n): " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        echo "Migration aborted."
        exit 1
    fi
fi

# Move subdirectories of multivariate
for subdir in "$OLD_MULTIVARIATE_DIR"/*; do
    if [ -d "$subdir" ]; then
        subdir_name=$(basename "$subdir")
        echo "Moving ${subdir_name} directory..."
        mkdir -p "${NEW_MULTIVARIATE_DIR}/${subdir_name}"
        if [ -d "${subdir}" ]; then
            cp -r "${subdir}"/* "${NEW_MULTIVARIATE_DIR}/${subdir_name}/"
            echo "  ${subdir_name} files copied to new location."
        fi
    elif [ -f "$subdir" ]; then
        # If there are any files directly in the multivariate directory
        file_name=$(basename "$subdir")
        echo "Moving file ${file_name}..."
        cp "${subdir}" "${NEW_MULTIVARIATE_DIR}/"
        echo "  ${file_name} copied to new location."
    fi
done

echo "Migration completed successfully."
echo "Old directory preserved at ${OLD_MULTIVARIATE_DIR}"
echo "New directory structure at ${NEW_MULTIVARIATE_DIR}"
echo ""
echo "If everything looks good, you can delete the old directory with:"
echo "rm -rf ${OLD_MULTIVARIATE_DIR}"

exit 0
