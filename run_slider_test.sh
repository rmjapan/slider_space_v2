#!/bin/bash
# Slider Testing Script
# Usage: ./run_slider_test.sh [concept_directory_name]

# Set default concept directory if not provided
CONCEPT_DIR=${1:-"concept_3253"}

# Base paths
BASE_DIR="/home/ryuichi/animins/slider_space_v2"
SLIDER_DIR="$BASE_DIR/trained_sliders/sdxl/$CONCEPT_DIR"
OUTPUT_DIR="$BASE_DIR/test_results/$CONCEPT_DIR"

# Check if slider directory exists
if [ ! -d "$SLIDER_DIR" ]; then
    echo "Error: Slider directory does not exist: $SLIDER_DIR"
    echo "Available concepts:"
    ls "$BASE_DIR/trained_sliders/sdxl/"
    exit 1
fi

echo "Testing sliders from: $SLIDER_DIR"
echo "Output will be saved to: $OUTPUT_DIR"
echo ""

# Navigate to the project directory
cd "$BASE_DIR"

# Run the test script
python test_sliders.py \
    --slider_dir "$SLIDER_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --base_prompt "1 girl, solo, masterpiece, best quality, character looking straight ahead, front view, head shot, general" \
    --device "cuda:0" \
    --num_sliders 10

echo ""
echo "Testing completed!"
echo "Check results in: $OUTPUT_DIR" 