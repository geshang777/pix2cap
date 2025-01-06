#!/bin/sh

## USAGE

## bash evaluation_VG.sh <path to the HF checkpointsh> <path to the directory to save the evaluation results> <path to test_caption.json> <path to images>


export PYTHONPATH="./:$PYTHONPATH"
export OMP_NUM_THREADS=1
MASTER_PORT=24997
NUM_GPUS=1  # Adjust it as per the available #GPU

# Check if the correct number of arguments is provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <ckpt_path> <result_path> <annotation_file> <image_dir>"
    exit 1
fi

# Positional arguments
CKPT_PATH=$1
RESULT_PATH=$2
ANNOTATION_FILE=$3
IMAGE_DIR=$4
DATASET=vg

# Run Inference
torchrun --nnodes=1 --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" eval/dense_caption/infer.py \
    --hf_model_path "$CKPT_PATH" \
    --annotation_file "$ANNOTATION_FILE" \
    --image_dir "$IMAGE_DIR" \
    --dataset "$DATASET" \
    --results_dir "$RESULT_PATH"

# Evaluate
python eval/dense_caption/evaluate.py \
    --annotation_file "$ANNOTATION_FILE" \
    --results_dir "$RESULT_PATH"