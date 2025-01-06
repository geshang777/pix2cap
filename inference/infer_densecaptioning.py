# ------------------------------------------------------------------------------------------------------
# Reference: https://github.com/microsoft/X-Decoder/blob/v2.0/inference/xdecoder/infer_panoseg.py
# Modified by Zuyao You (https://github.com/geshang777)
# ------------------------------------------------------------------------------------------------------

import os
import sys
import logging

pth = '/'.join(sys.path[0].split('/')[:-2])
sys.path.insert(0, pth)

from PIL import Image, ImageDraw, ImageFont
import numpy as np
np.random.seed(0)
import cv2
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

from utils.arguments import load_opt_command
from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks
from modeling.BaseModel import BaseModel
from modeling import build_model
from detectron2.utils.colormap import random_color
from utils.visualizer import Visualizer
from utils.distributed import init_distributed
import textwrap

logger = logging.getLogger(__name__)
def add_text_below_visimage(vis_image, text_list, font_size=20, padding=20):
    img = Image.fromarray(vis_image.get_image())
    
    img_width, img_height = img.size
    
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(img)

    bbox = draw.textbbox((0, 0), "Tg", font=font)
    line_height = bbox[3] - bbox[1]
    
    max_line_width = img_width - 2 * padding
    wrapped_lines = []
    
    for line in text_list:
        words = line.split()
        current_line = words[0]
        
        for word in words[1:]:
            bbox = draw.textbbox((0, 0), current_line + ' ' + word, font=font)
            line_width = bbox[2] - bbox[0]
            
            if line_width <= max_line_width:
                current_line += ' ' + word
            else:
                wrapped_lines.append(current_line)
                current_line = word
                
        wrapped_lines.append(current_line) 

    space_height = len(wrapped_lines) * line_height + 2 * padding
    
    new_img_height = img_height + space_height
    
    new_img = Image.new('RGB', (img_width, new_img_height), color=(255, 255, 255))
    
    new_img.paste(img, (0, 0))
    
    draw = ImageDraw.Draw(new_img)
    text_x = padding
    for i, line in enumerate(wrapped_lines):
        text_y = img_height + padding + i * line_height  
        draw.text((text_x, text_y), line, font=font, fill=(0, 0, 0))  
    
    
    
    return new_img

def main(args=None):
    '''
    Main execution point for PyLearn.
    '''
    
    opt, cmdline_args = load_opt_command(args)
    if cmdline_args.user_dir:
        absolute_user_dir = os.path.abspath(cmdline_args.user_dir)
        opt['base_path'] = absolute_user_dir
    opt = init_distributed(opt)

    # META DATA
    pretrained_pth = os.path.join(opt['RESUME_FROM'])
    # if 'novg' not in pretrained_pth:
    #     assert False, "Using the ckpt without visual genome training data will be much better."
    output_root = './output'
    image_pth = '/home/qid/pix2cap/xdecoder_data/coco/val2017/000000000632.jpg'
    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
    # model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(["background"], is_eval=False)

    t = []
    t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
    transform = transforms.Compose(t)
    thing_classes = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", 
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", 
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", 
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", 
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", 
        "toothbrush", "street"
    ]

    stuff_classes = [
        "banner", "blanket", "bridge", "cardboard", "counter", "curtain", "door-stuff", "floor-wood", 
        "flower", "fruit", "gravel", "house", "light", "mirror-stuff", "net", "pillow", "platform", 
        "playingfield", "railroad", "river", "road", "roof", "sand", "sea", "shelf", "snow", 
        "stairs", "tent", "towel", "wall-brick", "wall-stone", "wall-tile", "wall-wood", 
        "water-other", "window-blind", "window-other", "tree-merged", "fence-merged", 
        "ceiling-merged", "sky-other-merged", "cabinet-merged", "table-merged", "floor-other-merged", 
        "pavement-merged", "mountain-merged", "grass-merged", "dirt-merged", "paper-merged", 
        "food-other-merged", "building-other-merged", "rock-merged", "wall-other-merged", 
        "rug-merged"
    ]




    thing_colors = [random_color(rgb=True, maximum=255).astype(np.int32).tolist() for _ in range(len(thing_classes))]
    stuff_colors = [random_color(rgb=True, maximum=255).astype(np.int32).tolist() for _ in range(len(stuff_classes))]
    thing_dataset_id_to_contiguous_id = {x:x for x in range(len(thing_classes))}
    stuff_dataset_id_to_contiguous_id = {x+len(thing_classes):x for x in range(len(stuff_classes))}

    MetadataCatalog.get("demo").set(
        thing_colors=thing_colors,
        thing_classes=thing_classes,
        thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
        stuff_colors=stuff_colors,
        stuff_classes=stuff_classes,
        stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
    )
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(thing_classes + stuff_classes + ["background"], is_eval=False)
    metadata = MetadataCatalog.get('demo')
    model.model.metadata = metadata
    model.model.sem_seg_head.num_classes = len(thing_classes + stuff_classes)
  

    with torch.no_grad():
        image_ori = Image.open(image_pth).convert("RGB")
        width = image_ori.size[0]
        height = image_ori.size[1]
        image = transform(image_ori)
        image = np.asarray(image)
        image_ori = np.asarray(image_ori)
        images = torch.from_numpy(image.copy()).permute(2,0,1).cuda()

        batch_inputs = [{'image': images, 'height': height, 'width': width}]
        outputs = model.model.evaluate_dense_captioning(batch_inputs)
        visual = Visualizer(image_ori, metadata=metadata)

        pano_seg = outputs[-1]['panoptic_seg'][0]
        pano_seg_info = outputs[-1]['panoptic_seg'][1]

        for i in range(len(pano_seg_info)):
            if pano_seg_info[i]['category_id'] in metadata.thing_dataset_id_to_contiguous_id.keys():
                pano_seg_info[i]['category_id'] = metadata.thing_dataset_id_to_contiguous_id[pano_seg_info[i]['category_id']]
                pano_seg_info[i]['caption'] = pano_seg_info[i]['caption']
            else:
                pano_seg_info[i]['isthing'] = False
                pano_seg_info[i]['category_id'] = metadata.stuff_dataset_id_to_contiguous_id[pano_seg_info[i]['category_id']]
                pano_seg_info[i]['caption'] = pano_seg_info[i]['caption']
        demo,caption_pairs = visual.draw_dense_panoptic_seg(pano_seg.cpu(), pano_seg_info) # rgb Image
        print(caption_pairs)

        if not os.path.exists(output_root):
            os.makedirs(output_root)
        demo = add_text_below_visimage(demo,caption_pairs)
        demo.save(os.path.join(output_root, 'densemask.png'))


if __name__ == "__main__":
    main()
    sys.exit(0)