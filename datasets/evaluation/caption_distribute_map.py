# -------------------------------------------------------------------------------------------
# Reference: https://github.com/cocodataset/panopticapi/blob/master/panopticapi/evaluation.py
# Modified by Zuyao You (https://github.com/geshang777)
# -------------------------------------------------------------------------------------------
import argparse
from collections import defaultdict
from panopticapi.utils import get_traceback, rgb2id
import multiprocessing
import json
import time
import os
import numpy as np
import PIL.Image as Image
import os
import torch
from datasets.evaluation.eval.tokenizer.ptbtokenizer import PTBTokenizer
from datasets.evaluation.eval.bleu.bleu import Bleu
from datasets.evaluation.eval.meteor.meteor import Meteor
from datasets.evaluation.eval.rouge.rouge import Rouge
from datasets.evaluation.eval.cider.cider import Cider
from datasets.evaluation.eval.spice.spice import Spice
from nltk.translate import meteor_score
from tqdm import tqdm
OFFSET = 256 * 256 * 256
VOID = 0
def nltk_meteor(records, show_subprogressbar=True):
    scores = []
    if show_subprogressbar:
        iters = tqdm(records)
    else:
        iters = records
    for r in iters:
        generated_text = r["candidate"].split()
        reference_texts = [x.split() for x in r["references"]]
        if len(reference_texts)==0:
            reference_texts = [[""]]
        score = meteor_score.meteor_score(reference_texts, generated_text)
        scores.append(score)
    out = {}
    out["scores"] = scores
    out["average_score"] = sum(scores) / len(scores)
    return out

def caption_quality_multi_core(matched_annotations_list, gt_folder, pred_folder, categories):

    records = []
    gt_sum = 0
    for gt_ann, pred_ann in matched_annotations_list:
        pan_gt = np.array(Image.open(os.path.join(gt_folder, gt_ann['file_name'])), dtype=np.uint32)
        pan_gt = rgb2id(pan_gt)
        pan_pred = np.array(Image.open(os.path.join(pred_folder, pred_ann['file_name'])), dtype=np.uint32)
        pan_pred = rgb2id(pan_pred)
        gt_sum += len(gt_ann['segments_info'])


        gt_captions = {el['id']: el['description'] for el in gt_ann['segments_info']}
        pred_captions = {el['id']: el['caption'] for el in pred_ann['segments_info']}
        

        gt_segms = {el['id']: el for el in gt_ann['segments_info']}
        pred_segms = {el['id']: el for el in pred_ann['segments_info']}
       


        # predicted segments area calculation + prediction sanity checks
        pred_labels_set = set(el['id'] for el in pred_ann['segments_info'])
        
        labels, labels_cnt = np.unique(pan_pred, return_counts=True)
        for label, label_cnt in zip(labels, labels_cnt):
            if label not in pred_segms:
                if label == VOID:
                    continue
                raise KeyError('In the image with ID {} segment with ID {} is presented in PNG and not presented in JSON.'.format(gt_ann['image_id'], label))
            pred_segms[label]['area'] = label_cnt
            pred_labels_set.remove(label)
            if pred_segms[label]['category_id'] not in categories:
                raise KeyError('In the image with ID {} segment with ID {} has unknown category_id {}.'.format(gt_ann['image_id'], label, pred_segms[label]['category_id']))
        if len(pred_labels_set) != 0:
            raise KeyError('In the image with ID {} the following segment IDs {} are presented in JSON and not presented in PNG.'.format(gt_ann['image_id'], list(pred_labels_set)))
        
        # confusion matrix calculation
        pan_gt_pred = pan_gt.astype(np.uint64) * OFFSET + pan_pred.astype(np.uint64)
        gt_pred_map = {}
        labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
        for label, intersection in zip(labels, labels_cnt):
            gt_id = label // OFFSET
            pred_id = label % OFFSET
            gt_pred_map[(gt_id, pred_id)] = intersection
        
        # count all matched pairs
        gt_matched = set()
        pred_matched = set()
        gt_list = []
        pred_list = []
        max_iou_per_pred_label = {}
        for label_tuple, intersection in gt_pred_map.items():
            gt_label, pred_label = label_tuple
            if gt_label in gt_matched: 
                continue
            if gt_label not in gt_segms:
                continue
            if pred_label not in pred_segms:
                continue
            if gt_segms[gt_label]['iscrowd'] == 1:
                continue
            if gt_segms[gt_label]['category_id'] != pred_segms[pred_label]['category_id']:
                continue

            union = pred_segms[pred_label]['area'] + gt_segms[gt_label]['area'] - intersection - gt_pred_map.get((VOID, pred_label), 0)
            iou = intersection / union

            if pred_label not in max_iou_per_pred_label or iou > max_iou_per_pred_label[pred_label]['iou']:
                record = {
                    'iou': iou,
                    'candidate': gt_captions[int(gt_label)],
                    'references': pred_captions[int(pred_label)],
                    # 'category_id': gt_segms[gt_label]['category_id']
                }
                records.append(record)
    return records,gt_sum


def map_compute(gt_json_file, pred_json_file, gt_folder=None, pred_folder=None):

    start_time = time.time()
    with open(gt_json_file, 'r', encoding='utf-8') as f:
        gt_json = json.load(f)
    with open(pred_json_file, 'r', encoding='utf-8') as f:
        pred_json = json.load(f)

    if gt_folder is None:
        gt_folder = gt_json_file.replace('.json', '')
    if pred_folder is None:
        pred_folder = pred_json_file.replace('.json', '')
    categories = {el['id']: el for el in gt_json['categories']}

    print("Evaluation caption quality metrics:")
    print("Ground truth:")
    print("\tCaptions folder: {}".format(gt_folder))
    print("\tJSON file: {}".format(gt_json_file))
    print("Prediction:")
    print("\tCaptions folder: {}".format(pred_folder))
    print("\tJSON file: {}".format(pred_json_file))

    if not os.path.isdir(gt_folder):
        raise Exception("Folder {} with ground truth segmentations doesn't exist".format(gt_folder))
    if not os.path.isdir(pred_folder):
        raise Exception("Folder {} with predicted segmentations doesn't exist".format(pred_folder))

    pred_annotations = {el['image_id']: el for el in pred_json['annotations']}
    matched_annotations_list = []
    for gt_ann in gt_json['annotations']:
        image_id = gt_ann['image_id']
        if image_id not in pred_annotations:
            raise Exception('No prediction for the image with id: {}'.format(image_id))
        matched_annotations_list.append((gt_ann, pred_annotations[image_id]))

    records,npos = caption_quality_multi_core(matched_annotations_list, gt_folder, pred_folder, categories)
    min_overlaps = [0.3, 0.4, 0.5, 0.6, 0.7] # determine whether a predicted bounding box matches the ground-truth box
    min_scores = [-1, 0, 0.05, 0.1, 0.15, 0.2, 0.25]
    scores_raw = nltk_meteor(records)
    scores = scores_raw["scores"]
    ap_results = {}
    det_results = {}
    for min_overlap in min_overlaps:
        for min_score in min_scores:
            # go down the list and build tp,fp arrays
            n = len(records)
            tp = np.zeros(n)
            fp = np.zeros(n)
            for i in range(n):
                # pull up the relevant record
                r = records[i]
                if not r["references"]:
                    # nothing aligned to this predicted box in the ground truth
                    fp[i] = 1
                else:
                    if r['iou'] >= min_overlap and scores[i] > min_score:
                        tp[i] = 1
                    else:
                        fp[i] = 1
            fp = np.cumsum(fp) # Cumulative Sum. [1,1,1,1].cumsum() == [1,2,3,4]
            tp = np.cumsum(tp)
            rec = tp / npos # Recall
            prec = tp / (fp + tp) # Precision

            # compute max-interpolated average precision
            ap = 0
            apn = 0
            for t in np.arange(0, 1.01, 0.01):
                mask = rec >= t
                prec_masked = prec[mask]
                p = 0
                try:
                    p = np.max(prec_masked)
                except ValueError:
                    p = 0
                ap += p
                apn += 1
            ap /= apn

            # store it
            if min_score == -1:
                det_results[f"iou{min_overlap}"] = ap
            else:
                ap_results[f"iou{min_overlap}_score{min_score}"] = ap

    map = np.mean(list(ap_results.values()))
    detmap = np.mean(list(det_results.values()))

    # lets get out of here
    results = {
        "map": map,
        "ap_breakdown": ap_results,
        "detmap": detmap,
        "det_breakdown": det_results,
    }

    t_delta = time.time() - start_time

    print("Time elapsed: {:0.2f} seconds".format(t_delta))

    return results
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_json_file', type=str,
                        help="JSON file with ground truth data")
    parser.add_argument('--pred_json_file', type=str,
                        help="JSON file with predictions data")
    parser.add_argument('--gt_folder', type=str, default=None,
                        help="Folder with ground turth COCO format segmentations. \
                              Default: X if the corresponding json file is X.json")
    parser.add_argument('--pred_folder', type=str, default=None,
                        help="Folder with prediction COCO format segmentations. \
                              Default: X if the corresponding json file is X.json")
    args = parser.parse_args()
    caption_quality_compute(args.gt_json_file, args.pred_json_file, args.gt_folder, args.pred_folder)
