spi_datasets = [
    # STAGE 1
    ###############################################STAGE 1#############################################
    # {
    # 'type': 'coco_det',
    # 'vis_root': './data/coco',
    # },
    # {'type': 'RefCOCO',
    #  'ann_file': './data/mdetr_annotations/finetune_refcoco_train.json',
    #  'img_prefix': './data/coco_all/',
    #  },
    # {'type': 'RefCOCOP',
    #  'ann_file': './data/mdetr_annotations/finetune_refcoco+_train.json',
    #  'img_prefix': './data/coco_all/',
    #  },

    ###############################################STAGE 2#############################################
    # STAGE 2

    {"type": "RefCOCOG",
     'ann_file': './data/mdetr_annotations/finetune_refcocog_train.json',
     'img_prefix': './data/coco_imgs/',
     },
    {"type": "flickr30k",
     'ann_file': './data/mdetr_annotations/final_flickr_mergedGT_train.json',
     'img_prefix': './data/flickr30k-images/',
     },

    {"type": "VGDATA",
     'ann_file': './data/visual_genome/train.json',
     'img_prefix': './data/visual_genome/vg_all',

     },

    {"type": "DENSECOCO",
     'ann_file': 'path/to/panoptic_train2017_densecap_v3.json',
     'img_prefix': 'path/to/coco2017/train2017',

     },

    {
        "type": "det_llava",
        "data_path": "./data/coco_imgs/",
        "ann_path": "./data/llava/llava_instruct_150k.json",
        "det_pkl_path": "./data/llava/llava_150k_bbox_pred_results.pkl"
    },
    {
        "type": "vcr",
        "ann_file": "./data/vcr/train.jsonl",
        "img_prefix": "./data/vcr/vcr1images"
    },
    {
        "type": "single_vcr",
        "ann_file": "./data/vcr/train.jsonl",
        "img_prefix": "./data/vcr/vcr1images"
    },
    {
        "type": "multi_vcr",
        "ann_file": "./data/vcr/train.jsonl",
        "img_prefix": "./data/vcr/vcr1images"
    }
]



"""{
    "type": "vcr",
    "ann_file": "./data/vcr/train.jsonl",
    "img_prefix": "./data/vcr/vcr1images"
},
{
    "type": "single_vcr",
    "ann_file": "./data/vcr/train.jsonl",
    "img_prefix": "./data/vcr/vcr1images"
},
{
    "type": "multi_vcr",
    "ann_file": "./data/vcr/train.jsonl",
    "img_prefix": "./data/vcr/vcr1images"
}"""