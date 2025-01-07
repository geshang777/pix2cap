import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from functools import partial
import copy

from llava import LlavaLlamaForCausalLM
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
import re
from PIL import Image
import random
import math
from gpt4roi.train.train import preprocess, preprocess_multimodal

multimodal_cfg = {'is_multimodal': True,
                  'sep_image_conv_front': False,
                  'image_token_len': 256,
                  'image_aspect_ratio': 'square',
                  'use_im_start_end': True}


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def patch_config(config):
    patch_dict = {
        "use_mm_proj": True,
        "mm_vision_tower": "openai/clip-vit-large-patch14",
        "mm_hidden_size": 1024
    }

    cfg = AutoConfig.from_pretrained(config)
    if not hasattr(cfg, "mm_vision_tower"):
        print(f'`mm_vision_tower` not found in `{config}`, applying patch and save to disk.')
        for k, v in patch_dict.items():
            setattr(cfg, k, v)
        cfg.save_pretrained(config)


def eval_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    from gpt4roi.models.spi_llava import SPILlavaMPTForCausalLM

    model = SPILlavaMPTForCausalLM.from_pretrained(model_name,
                                                   low_cpu_mem_usage=True,
                                                   torch_dtype=torch.float16,
                                                   use_cache=True).cuda()

    # model = SPILlavaMPTForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, use_cache=True)
    
    image_processor = CLIPImageProcessor.from_pretrained(
        "openai/clip-vit-large-patch14", torch_dtype=torch.float16)

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
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    ans_file = open(answers_file, "w")
    for i, line in enumerate(tqdm(questions)):
        idx = line["question_id"]
        image_file = line["image"]
        obj_boxes = line['obj_boxes']
        qs = line["text"]
        objs = re.findall(r'<(obj\d*)>', qs)
        boxes = []
        for obj in objs:
            boxes.append(obj_boxes[obj])
            obj_index = int(re.search(r'\d+', obj).group()) if re.search(r'\d+', obj) else 1
            qs = qs.replace(f'<{obj}>', f'`region{obj_index} <bbox>`')
        boxes = torch.tensor(boxes).cuda().half()
        width = line['width']
        height = line['height']
        bboxes = boxes / torch.tensor([width, height, width, height]).cuda().half()


        image = Image.open(os.path.join(args.image_folder, image_file))
        image = image_processor.preprocess(image,
                                        do_center_crop=False,
                                        return_tensors='pt')['pixel_values'][0]

        image = torch.nn.functional.interpolate(image.unsqueeze(0),
                                                size=(224, 224),
                                                mode='bilinear',
                                                align_corners=False).squeeze(0)

        cur_token_len = (image.shape[1] // 14) * (image.shape[2] // 14)  # FIXME: 14 is hardcoded patch size

        begin_str = "The <image> provides an overview of the picture.\n"

        init_question = begin_str + qs
        sources = dict()
        sources['conversations'] = []
        sources['conversations'].append(
            {'from': 'human', 'value': init_question})
        sources = preprocess_multimodal([sources['conversations']],
                                        multimodal_cfg, cur_token_len)
        ori_source = copy.deepcopy(sources)

        data_dict = preprocess(
            sources,
            tokenizer)

        data_dict = dict(input_ids=data_dict['input_ids'][0],
                        labels=data_dict['labels'][0],
                        sources=ori_source,
                        init_question=init_question,
                        )
        input_ids = data_dict['input_ids'].cuda()[None]

        # new stopping implementation
        class KeywordsStoppingCriteria(StoppingCriteria):
            def __init__(self, keywords, tokenizer, input_ids):
                self.keywords = keywords
                self.tokenizer = tokenizer
                self.start_len = None
                self.input_ids = input_ids

            def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                if self.start_len is None:
                    self.start_len = self.input_ids.shape[1]
                else:
                    outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
                    for keyword in self.keywords:
                        if keyword in outputs:
                            return True
                return False
        stop_str = '###'
        keywords = ['###']
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
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
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] Sample {i}: {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]

        if args.conv_mode == 'simple_legacy' or args.conv_mode == 'simple':
            while True:
                cur_len = len(outputs)
                outputs = outputs.strip()
                for pattern in ['###', 'Assistant:', 'Response:']:
                    if outputs.startswith(pattern):
                        outputs = outputs[len(pattern):].strip()
                if len(outputs) == cur_len:
                    break

        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.split(': ', 1)[-1].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": init_question,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--mm-projector", type=str, default=None)
    parser.add_argument("--vision-tower", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="simple")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()

    eval_model(args)
