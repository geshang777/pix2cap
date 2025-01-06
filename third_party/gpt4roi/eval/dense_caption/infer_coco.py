import argparse
import copy
import os
from functools import partial
from io import BytesIO

import matplotlib.pyplot as plt
import requests
import torch
from matplotlib.patches import Rectangle
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel

import mmcv
from gpt4roi.train.train import preprocess, preprocess_multimodal
from llava.model.utils import KeywordsStoppingCriteria
from llava.utils import disable_torch_init

import os
import torch.distributed as dist

import subprocess
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from eval.utils import bbox_to_x1y1x2y2
from torch.utils.data import DataLoader, DistributedSampler
from eval.transforms import ResizeLongestSide
from tqdm import tqdm
import json
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser(description="GLaMM Inference - Region Captioning")

    parser.add_argument("--hf_model_path", required=True, help="The model path in huggingface format.")
    parser.add_argument("--annotation_file",
                        default="/share_io03_ssd/test2/youzuyao/gpt4roi/data/visual_genome/test_caption.json", type=str,
                        help="Replace with 'data/visual_genome/test_caption.json' for VG.")
    parser.add_argument("--image_dir", default="share_io03_ssd/test2/konglingyu/VG/images", type=str,
                        help="Replace with 'data/visual_genome/images' for VG")
    parser.add_argument("--dataset", default="refcocog", type=str, help="Options are 'refcocog', 'vg'")
    parser.add_argument("--results_dir", default="results", type=str, help="The path to save the results.")


    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"], )

    # DDP Related parameters
    parser.add_argument("--batch_size_per_gpu", required=False, default=1)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser.parse_args()

class RegionCapDDP(Dataset):
    def __init__(self, annotation_file):
        self.coco = COCO(annotation_file)
        self.image_dict = self.coco.imgs
        self.ann_dict = self.coco.anns
        self.image_dict_keys = list(self.image_dict.keys())
        self.ann_dict_keys = list(self.ann_dict.keys())


    def __len__(self):
        # return len(self.image_dict_keys)
        return len(self.ann_dict_keys)

    def __getitem__(self, idx):
        # image_id = self.image_dict_keys[idx]
        # filename = self.image_dict[image_id]['file_name']
        # bbox = bbox_to_x1y1x2y2(self.ann_dict[image_id]['bbox'])
        # gt = self.ann_dict[image_id]['caption']

        image_id = self.ann_dict_keys[idx]
        filename = self.ann_dict[image_id]['file_name'].replace(".png",".jpg")
        bbox = bbox_to_x1y1x2y2(self.ann_dict[image_id]['bbox'])
        gt = self.ann_dict[image_id]['description']

        return image_id, filename, bbox, gt
    

DEFAULT_IMAGE_TOKEN = '<image>'
DEFAULT_IMAGE_PATCH_TOKEN = '<im_patch>'
DEFAULT_IM_START_TOKEN = '<im_start>'
DEFAULT_IM_END_TOKEN = '<im_end>'

multimodal_cfg = {'is_multimodal': True,
                  'sep_image_conv_front': False,
                  'image_token_len': 256,
                  'image_aspect_ratio': 'square',
                  'use_im_start_end': True}


def load_image(image_file):

    image = Image.open(image_file).convert('RGB')
    return image


def get_init_inputs(img_path,
                    processor,
                    tokenizer,
                    bbox,
):
    # det_model = build_det_model_from_cfg()
    # bbox_results = inf_single_image(det_model, img_path, thr=0.3, number=100)
    image = load_image(img_path)
    width, height = image.size
    ori_bboxes = np.array(bbox, dtype=np.float64)
    norm_bboxes = ori_bboxes / np.array([width, height, width, height])
    image = processor.preprocess(image,
                                    do_center_crop=False,
                                    return_tensors='pt')['pixel_values'][0]

    image = torch.nn.functional.interpolate(image.unsqueeze(0),
                                            size=(224, 224),
                                            mode='bilinear',
                                            align_corners=False).squeeze(0)

    cur_token_len = (image.shape[1] // 14) * (image.shape[2] // 14)  # FIXME: 14 is hardcoded patch size

    # pred_bboxes = bbox_results
    # ori_bboxes = pred_bboxes

    # w, h = pred_bboxes[:, 2] - pred_bboxes[:, 0], pred_bboxes[:, 3] - pred_bboxes[:, 1]
    # filter_small = (w > 0.02) & (h > 0.02)
    # pred_bboxes = pred_bboxes[filter_small]
    # if len(pred_bboxes) == 0:
    #     pred_bboxes = ori_bboxes[:10][:, :4]
    # begin_str = 'The <image> describes the entire picture, while <spi_descript> describes specific regions within the image.\n'
    # print('please input the question:')
    # question_str = input()
    # question_str = "debug"

    begin_str = "The <image> provides an overview of the picture.\n"

    init_question = begin_str + "Can you give a description of the region mentioned by `region1 <bbox>`?"#begin_str + question_str

#     begin_str = "The <image> provides an overview of the picture.\n"

#     init_question = begin_str + "Can you provide me with a detailed description of the region in the picture marked by <spi_descript>?"
# #begin_str + question_str

    # init_question = init_question.replace('<spi_descript>', '<bbox>' * len(bbox))
    sources = dict()
    sources['conversations'] = []
    sources['conversations'].append(
        {'from': 'human', 'value': init_question})
    sources = preprocess_multimodal([sources['conversations']],
                                    multimodal_cfg, cur_token_len)
    ori_source = copy.deepcopy(sources)

    # import pdb; pdb.set_trace()
    data_dict = preprocess(
        sources,
        tokenizer)

    data_dict = dict(input_ids=data_dict['input_ids'][0],
                     labels=data_dict['labels'][0],
                     sources=ori_source,
                     init_question=init_question,
                     )

    data_dict['image'] = image

    data_dict['bboxes'] = torch.Tensor(norm_bboxes)

    data_dict['img_metas'] = dict(filename=img_path)

    return data_dict


def vis(img_path, gt, pred, bboxes=None, region_cap=None, id=0, dir='coco'):
    img = Image.open(img_path)

    fig, ax = plt.subplots()
    width = img.width
    height = img.height
    ax.imshow(img)
    if bboxes is not None:
        for r_id, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            rect = Rectangle((x1 * width, y1 * height), w * width, h * height,
                             linewidth=5, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            if region_cap:
                text = region_cap[r_id]  # 根据需要修改标注的文字
                ax.text(x1, y1, text, fontsize=10, color='blue')

    ax.text(0, -20, f'gt:{gt}', fontsize=6, color='red')
    ax.text(0, -10, f'pred:{pred}', fontsize=6, color='blue')
    plt.savefig(f'{dir}/{img_path.split("/")[-1]}_{id}.jpg')
def custom_collate_fn(batch):
    image_id = [item[0] for item in batch]
    filename = [item[1] for item in batch]
    bbox = [item[2] for item in batch]
    gt = [item[3] for item in batch]

    return image_id, filename, bbox, gt

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model-name', type=str,
    #                     default='/share_io03_ssd/test2/xieyiweng/debug')
    # # parser.add_argument("--det", type=str, default="eva2_o365_to_lvis_cascade_mask_rcnn_vitdet_l_8attn_1536_lrd0p8.py")
    # parser.add_argument('--img', type=str, default='/vhome/youzuyao/COD/1.jpgg')
    # args = parser.parse_args()
    args = parse_args()




    disable_torch_init()
    model_name = os.path.expanduser(args.hf_model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    from gpt4roi.models.spi_llava import SPILlavaMPTForCausalLM

    model = SPILlavaMPTForCausalLM.from_pretrained(model_name,
                                                   low_cpu_mem_usage=True,
                                                   torch_dtype=torch.float16,
                                                   use_cache=True).cuda()

    # model = SPILlavaMPTForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, use_cache=True)
    image_processor = CLIPImageProcessor.from_pretrained(
        model.config.mm_vision_tower, torch_dtype=torch.float16)

    mm_use_im_start_end = getattr(model.config, 'mm_use_im_start_end', False)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN],
                         special_tokens=True)
    spi_tokens = ['<bbox>', '<point>']
    tokenizer.add_tokens(spi_tokens, special_tokens=True)

    vision_tower = model.get_model().vision_tower[0]

    if vision_tower.device.type == 'meta':
        vision_tower = CLIPVisionModel.from_pretrained(
            vision_tower.config._name_or_path, torch_dtype=torch.float16,
            low_cpu_mem_usage=True).cuda()
        model.get_model().vision_tower[0] = vision_tower
    else:
        vision_tower.to(device='cuda', dtype=torch.float16)

    vision_tower.to(device='cuda', dtype=torch.float16)
    vision_config = vision_tower.config
    vision_config.im_patch_token = \
        tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end

    vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids(
        [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    transform = ResizeLongestSide(args.image_size)

    model.eval()
    args.rank = int(os.environ["RANK"])

    dist.init_process_group(backend='nccl', init_method='env://')

    # Create DDP Dataset
    # instruction = "Can you provide me with a detailed description of the region in the picture marked by <bbox>?"
    instruction = "In the conversation below, you simply answer the category name based on what you see in the imagery inside a particular region. I will give you only one region each time. Categories containing person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush, street"
    dataset = RegionCapDDP(args.annotation_file)
    distributed_sampler = DistributedSampler(dataset, rank=args.rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size_per_gpu, num_workers=2,
                            sampler=distributed_sampler, collate_fn=custom_collate_fn)

    # Iterate over all the samples, perform inference and save results
    results_path = f"{args.results_dir}/{os.path.basename(args.hf_model_path)}_{args.dataset}_{args.rank}.json"

    results = []
    for idx, (image_id, filename, bbox, gt) in enumerate(tqdm(dataloader)):
        image_id, filename, bbox, gt = image_id[0], filename[0], bbox[0], gt[0]
        image_path = os.path.join(args.image_dir, filename)
        init_inputs = get_init_inputs(image_path,
                                      image_processor,
                                      tokenizer,
                                      [bbox],
                                      )
        bboxes = init_inputs['bboxes'].cuda()
        image = init_inputs['image']
        input_ids = init_inputs['input_ids'].cuda()[None]

        stop_str = '###'
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer,
                                                     input_ids)

        model.model.tokenizer = tokenizer

        with torch.inference_mode():

            model.orig_forward = model.forward
            model.forward = partial(model.orig_forward,
                                    img_metas=[None],
                                    bboxes=[bboxes.half()])
            with torch.amp.autocast(device_type='cuda'):
                output_ids = model.generate(
                    input_ids,
                    images=image.unsqueeze(0).half().cuda(),
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=1024,
                    stopping_criteria=[stopping_criteria])
            model.forward = model.orig_forward

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (
                input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:],
                                         skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.split(': ', 1)[-1].strip()
        print(f"Image ID: {image_id}, Caption: {outputs}, GT: {gt}")
        # img = Image.open(image_path)
        # inputs = {'image': img, 'boxes': [bbox]}

        # result_caption = inference(instruction, inputs)  # Perform inference

        result_dict = {}
        result_dict["image_id"] = image_id
        result_dict["caption"] = outputs #result_caption
        
        results.append(result_dict)

    with open(results_path, 'w') as json_file:
        json.dump(results, json_file, indent=2)