import os
import json
import argparse
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


def parse_args():
    parser = argparse.ArgumentParser(description="GLaMM Inference - Region Captioning")

    parser.add_argument("--annotation_file",
                        default="/share_io03_ssd/test2/youzuyao/gpt4roi/data/visual_genome/test_caption.json", type=str,
                        help="Replace with 'data/visual_genome/test_caption.json' for VG.")
    parser.add_argument("--results_dir", default="/vhome/youzuyao/gpt4roi/eval/vg_result", type=str, help="The path to save the results.")

    return parser.parse_args()


def main():
    args = parse_args()

    # Load the annotation file
    coco = COCO(args.annotation_file)

    # Merge and load the results files
    all_results = []
    for result_file in os.listdir(args.results_dir):
        all_results += json.load(open(f"{args.results_dir}/{result_file}", "r"))

    merged_file_path = f"{args.results_dir}/merged.json"
    with open(merged_file_path, 'w') as f:
        json.dump(all_results, f)
    coco_result = coco.loadRes(merged_file_path)

    # Create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # Evaluate results
    coco_eval.params['image_id'] = coco_result.getImgIds()
    coco_eval.evaluate()

    # Print and save the output evaluation scores
    output_file_path = f"{args.results_dir}/metrics.txt"
    f = open(output_file_path, 'w')
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')
        f.write(f"{metric}: {score:.3f}\n")
    f.close()


if __name__ == "__main__":
    main()