model_name=gpt4roi_plus_pix2cap
model_path=mucai/$model_name
folder=ViP-Bench
split=$1
mkdir -p ./playground/data/eval/$folder/results

python scripts/convert_vipbench_for_eval.py \
    --src ./playground/data/eval/$folder/answers/$model_name-$split.jsonl \
    --dst ./playground/data/eval/$folder/results/$model_name-$split.json