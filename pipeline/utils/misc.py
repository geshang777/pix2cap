import logging
import torch

logger = logging.getLogger(__name__)

def hook_opt(opt):

    try:
        grounding_flag = opt['REF']['INPUT']['SPATIAL']
    except:
        grounding_flag = False

    if grounding_flag:
        opt['ATTENTION_ARCH']['SELF_ATTENTION']['queries']['grounding'] = ['queries_grounding', 'tokens_grounding', 'tokens_spatial']

    try:
        spatial_flag = opt['STROKE_SAMPLER']['EVAL']['GROUNDING']
    except:
        spatial_flag = False

    if spatial_flag:
        opt['ATTENTION_ARCH']['SELF_ATTENTION']['queries']['spatial'] = ['queries_spatial', 'tokens_spatial', 'memories_spatial', 'tokens_grounding']

    return opt

# HACK for evalution 
def hook_metadata(metadata, name):
    return metadata

# HACK for evalution 
def hook_switcher(model, name):
    mappings = {'SEMANTIC_ON': True, 'INSTANCE_ON': True, 'PANOPTIC_ON': True}

    for key, value in mappings.items():
        if key == 'SEMANTIC_ON':
            model.model.semantic_on = value
        if key == 'INSTANCE_ON':
            model.model.instance_on = value
        if key == 'PANOPTIC_ON':
            model.model.panoptic_on = value