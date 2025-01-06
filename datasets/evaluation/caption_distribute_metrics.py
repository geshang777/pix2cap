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
from datasets.evaluation.eval.tokenizer.ptbtokenizer import PTBTokenizer
from datasets.evaluation.eval.bleu.bleu import Bleu
from datasets.evaluation.eval.meteor.meteor import Meteor
from datasets.evaluation.eval.rouge.rouge import Rouge
from datasets.evaluation.eval.cider.cider import Cider
from datasets.evaluation.eval.spice.spice import Spice
import logging
logger = logging.getLogger(__name__)

OFFSET = 256 * 256 * 256
VOID = 0
class CaptionQualityStatCat:
    def __init__(self):

        self.n = 0
        self.iou = 0.0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.gt_cap = {}
        self.pred_cap = {}

    def __iadd__(self, other):

        self.n += other.n
        self.iou += other.iou
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.gt_cap.update(other.gt_cap)
        self.pred_cap.update(other.pred_cap)
        return self

    def is_empty(self):
        return self.n == 0
class CaptionQualityStat:
    def __init__(self):
        self.quality_per_cat = defaultdict(CaptionQualityStatCat)

    def __getitem__(self, i):
        return self.quality_per_cat[i]

    def __iadd__(self, quality_stat):
        for label, quality_stat_cat in quality_stat.quality_per_cat.items():
            self.quality_per_cat[label] += quality_stat_cat
        return self

    def quality_average(self, categories,isthing):
        bleu, meteor, rouge, cider, spice, n = 0, 0, 0, 0, 0, 0
        pq, sq, rq = 0, 0, 0
        total = 0
        per_class_results = {}
        gt_cap={}
        pred_cap={}
        tokenizer = PTBTokenizer()
        total_fp, total_tp, total_fn = 0, 0, 0
        for label, label_info in categories.items():
            if label not in self.quality_per_cat:
                continue
            if isthing is not None:
                cat_isthing = label_info['isthing'] == 1
                if isthing != cat_isthing:
                    continue

            count = self.quality_per_cat[label].n

            iou = self.quality_per_cat[label].iou
            tp = self.quality_per_cat[label].tp
            fp = self.quality_per_cat[label].fp
            fn = self.quality_per_cat[label].fn
            total = total + tp +fn + fp
            total_fp += fp
            total_tp += tp
            total_fn += fn
            
            gt_cap.update(self.quality_per_cat[label].gt_cap)
            pred_cap.update(self.quality_per_cat[label].pred_cap)
            if tp + fp + fn == 0:
                per_class_results[label] = {'pq': 0.0, 'sq': 0.0, 'rq': 0.0}
                continue

            n += 1
            pq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
            sq_class = iou / tp if tp != 0 else 0
            rq_class = tp / (tp + 0.5 * fp + 0.5 * fn)
            per_class_results[label] = {'pq': pq_class, 'sq': sq_class, 'rq': rq_class}

            pq += pq_class
            sq += sq_class
            rq += rq_class
        logger.info("tp:{},fn:{},fp:{}".format(total_tp,total_fn,total_fp))
        logger.info("total:{},gt:{}".format(total,len(gt_cap)))
        gts = tokenizer.tokenize(gt_cap)
        res = tokenizer.tokenize(pred_cap)
        scorers = []
        scorers.append((Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]))
        scorers.append((Rouge(), "ROUGE_L"))
        scorers.append((Cider(), "CIDEr"))
        scorers.append((Spice(), "SPICE"))
        # scorers.append((Meteor(), "METEOR"))


        for scorer, method in scorers:
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                bleu = score[3]*len(gts)/total

            else:
                if method == "CIDEr":
                    cider = score*len(gts)/total
                elif method == "METEOR":
                    meteor = score*len(gts)/total
                elif method == "ROUGE_L":
                    rouge = score*len(gts)/total
                elif method == "SPICE":
                    spice = score*len(gts)/total
            logger.info(f"{method}: {score}")
        
        return {'bleu': bleu , 'meteor': meteor , 'cider': cider,'rouge': rouge,'spice':spice, 'pq': pq / n, 'sq': sq / n, 'rq': rq / n, 'n': n}#, per_class_results

    def is_empty(self):
        return all(cat_stat.is_empty() for cat_stat in self.quality_per_cat.values())
def caption_quality_single_core(proc_id, annotation_set, gt_folder, pred_folder, categories):
    quality_stat = CaptionQualityStat()
    idx = 0
    num_captions = 0
    for gt_ann, pred_ann in annotation_set:
        if idx % 100 == 0:
            print('Core: {}, {} from {} images processed'.format(proc_id, idx, len(annotation_set)))
        idx += 1

        pan_gt = np.array(Image.open(os.path.join(gt_folder, gt_ann['file_name'])), dtype=np.uint32)
        pan_gt = rgb2id(pan_gt)
        pan_pred = np.array(Image.open(os.path.join(pred_folder, pred_ann['file_name'])), dtype=np.uint32)
        pan_pred = rgb2id(pan_pred)


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

        # match gt and pred by iou
        for label_tuple, intersection in gt_pred_map.items():
            gt_label, pred_label = label_tuple
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
            if iou > 0.5:
                quality_stat[gt_segms[gt_label]['category_id']].tp += 1
                quality_stat[gt_segms[gt_label]['category_id']].iou += iou
                gt_matched.add(gt_label)
                pred_matched.add(pred_label)
                gt_list.append(gt_label)
                pred_list.append(pred_label)
           
        # count false negtives
        crowd_labels_dict = {}
        for gt_label, gt_info in gt_segms.items():
            if gt_label in gt_matched:
                continue
            # crowd segments are ignored
            if gt_info['iscrowd'] == 1:
                crowd_labels_dict[gt_info['category_id']] = gt_label
                continue
            quality_stat[gt_info['category_id']].fn += 1



        # count false positives
        for pred_label, pred_info in pred_segms.items():
            if pred_label in pred_matched:
                continue
            # intersection of the segment with VOID
            intersection = gt_pred_map.get((VOID, pred_label), 0)
            if pred_info['category_id'] in crowd_labels_dict:
                intersection += gt_pred_map.get((crowd_labels_dict[pred_info['category_id']], pred_label), 0)
            if intersection / pred_info['area'] > 0.5:
                continue
            quality_stat[pred_info['category_id']].fp += 1
            # num_captions+=1


        for idx,seg_id in enumerate(gt_list):

            pred_caption = pred_captions[pred_list[idx]]
            if isinstance(pred_caption, list):
                continue
            gt_caption = gt_captions[seg_id]
            image_id = str(gt_ann['image_id'])+ '_' + str(int(seg_id))

            category_id = pred_segms[pred_list[idx]]['category_id']
            quality_stat[category_id].gt_cap[image_id] = [{"caption":gt_caption}]
            quality_stat[category_id].pred_cap[image_id] = [{"caption":pred_caption}]

            quality_stat[category_id].n += 1
           

    print('Core: {}, all {} images processed'.format(proc_id, len(annotation_set)))

    return quality_stat #,num_captions

def caption_quality_multi_core(matched_annotations_list, gt_folder, pred_folder, categories):
    cpu_num = multiprocessing.cpu_count()
    annotations_split = np.array_split(matched_annotations_list, cpu_num)
    print("Number of cores: {}, images per core: {}".format(cpu_num, len(annotations_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, annotation_set in enumerate(annotations_split):
        p = workers.apply_async(caption_quality_single_core,
                                (proc_id, annotation_set, gt_folder, pred_folder, categories))
        processes.append(p)
    quality_stat = CaptionQualityStat()
    for p in processes:
        result = p.get()
        if not result.is_empty():
            quality_stat += result
        else:
            print(f'Process {proc_id} returned an empty result.')
    return quality_stat

def caption_quality_compute(gt_json_file, pred_json_file, gt_folder=None, pred_folder=None):
    
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

    quality_stat = caption_quality_multi_core(matched_annotations_list, gt_folder, pred_folder, categories)
    metrics = [("All", None)]
    results = {}
    for name, isthing in metrics:
        results[name] = quality_stat.quality_average(categories, isthing=isthing)

    # results, per_class_results = quality_stat.quality_average(categories)
    print("{:10s}| {:>5s}  {:>5s}  {:>5s}  {:>5s} {:>5s}  {:>5s}  {:>5s}  {:>5s}  {:>5s}".format("", "BLEU", "CIDER", "METEOR","ROUGE","SPICE" ,"PQ", "SQ", "RQ", "N"))
    print("-" * (10 + 7 * 9))
    for name, _isthing in metrics:
        print("{:10s}| {:5.1f}  {:5.1f}  {:5.1f}  {:5.1f} {:5.1f}  {:5.1f}  {:5.1f}  {:5.1f}  {:5d}".format(
            name,
            100 *results[name]['bleu'],
            100 *results[name]['cider'],
            100 *results[name]['meteor'],
            100 *results[name]['rouge'],
            100 *results[name]['spice'],
            100 * results[name]['pq'],
            100 * results[name]['sq'],
            100 * results[name]['rq'],
            results[name]['n'])
        )

    t_delta = time.time() - start_time

    print("Time elapsed: {:0.2f} seconds".format(t_delta))

    return results#, per_class_results

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
