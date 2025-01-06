# ------------------------------------------------------------------------------------------------------
# Reference: https://github.com/microsoft/X-Decoder/blob/v2.0/modeling/interface/xdecoder.py
# Modified by Zuyao You (https://github.com/geshang777)
# ------------------------------------------------------------------------------------------------------

import logging
from typing import Optional
import string
import torch
import numpy as np
from torch import nn, Tensor
from torch.nn import functional as F
from transformers import BertTokenizer
from detectron2.modeling.roi_heads.cascade_rcnn import _ScaleGradient

from timm.models.layers import trunc_normal_
from detectron2.layers import Conv2d
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as nnf

from .build import register_decoder
from .modules import SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP
from ..utils import configurable
from ..modules import PositionEmbeddingSine
import torch
from ..text.text_decoder import TransformerDecoderTextualHead, GRiTTextDecoder, AutoRegressiveBeamSearch
from ..text.load_text_token import LoadTextTokens
import torchvision.ops.boxes as box_ops
import random
from ..utils.box_ops import box_cxcywh_to_xyxy

logger = logging.getLogger(__name__)
class Pix2Cap_Transformer_Decoder(nn.Module):

    @configurable
    def __init__(
        self,
        lang_encoder: nn.Module,
        in_channels,
        mask_classification=True,
        *,
        hidden_dim: int,
        dim_proj: int,
        num_queries: int,
        contxt_len: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        task_switch: dict,
        enforce_input_project: bool,
        dataset: str,
        mask_head_enabled: bool,
        object_feature_size: int,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        attention_heads: int,
        feedforward_size: int,
        mask_future_positions: bool,
        padding_idx: int,
        decoder_type: str,
        use_act_checkpoint: bool,
        max_steps: int,
        beam_size: int,

    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()
        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification
        self.dataset = dataset.split('_')[0]
        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.contxt_len = contxt_len
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.mask_head_enabled = mask_head_enabled


        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )


        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        self.task_switch = task_switch

        # output FFNs
        self.lang_encoder = lang_encoder
        if self.task_switch['mask']:
            self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        self.class_embed = nn.Parameter(torch.empty(hidden_dim, dim_proj))
        trunc_normal_(self.class_embed, std=.02)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # Caption Project and query
        # if task_switch['captioning']:
        self.caping_embed = nn.Parameter(torch.empty(hidden_dim, dim_proj))
        trunc_normal_(self.caping_embed, std=.02)
        self.pos_embed_caping = nn.Embedding(contxt_len, hidden_dim)

        # register self_attn_mask to avoid information leakage, it includes interaction between object query, class query and caping query
        self_attn_mask = torch.zeros((1, num_queries + contxt_len, num_queries + contxt_len)).bool()
        self_attn_mask[:, :num_queries, num_queries:] = True # object+class query does not attend with caption query.
        self_attn_mask[:, num_queries:, num_queries:] = torch.triu(torch.ones((1, contxt_len, contxt_len)), diagonal=1).bool() # caption query only attend with previous token.
        self_attn_mask[:, :num_queries-1, num_queries-1:num_queries] = True # object query does not attend with class query.
        self_attn_mask[:, num_queries-1:num_queries, :num_queries-1] = True # class query does not attend with object query.
        self.register_buffer("self_attn_mask", self_attn_mask)
        # self.frozen_stages = 1

        #grit
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.tokenizer = tokenizer
        self.get_target_text_tokens = LoadTextTokens(tokenizer, max_text_len=40, padding='do_not_pad')
        text_decoder_transformer = TransformerDecoderTextualHead(
            object_feature_size=object_feature_size,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            attention_heads=attention_heads,
            feedforward_size=feedforward_size,
            mask_future_positions=mask_future_positions,
            padding_idx=padding_idx,
            decoder_type=decoder_type,
            use_act_checkpoint=use_act_checkpoint,
        )
        test_task = "DenseCap"
        task_begin_tokens = {}
        train_task = ["DenseCap"]
        for i, task in enumerate(train_task):
            if i == 0:
                task_begin_tokens[task] = tokenizer.cls_token_id
            else:
                task_begin_tokens[task] = 103 + i
        self.task_begin_tokens = task_begin_tokens
        beamsearch_decode = AutoRegressiveBeamSearch(
            end_token_id=tokenizer.sep_token_id,
            max_steps=40,
            beam_size=beam_size,
            objectdet=False,
            per_node_beam_size=1,
        )
        self.text_decoder = GRiTTextDecoder(
            text_decoder_transformer,
            beamsearch_decode=beamsearch_decode,
            begin_token_id=task_begin_tokens[test_task],
            loss_type='smooth',
            tokenizer=tokenizer,
        )
        self.object_proj=nn.Linear(hidden_dim, object_feature_size)
        self.global_pooling = nn.AdaptiveMaxPool1d(1)

        

    def _activate_stages(self):
        if self.mask_head_enabled:
            for layer in self.transformer_self_attention_layers:
                for param in layer.parameters():
                    param.requires_grad = True

            for layer in self.transformer_cross_attention_layers:
                for param in layer.parameters():
                    param.requires_grad = True
            for layer in self.transformer_ffn_layers:
                for param in layer.parameters():
                    param.requires_grad = True
            for param in self.mask_embed.parameters():
                param.requires_grad = True
        for param in self.text_decoder.parameters():
            param.requires_grad = True
        for param in self.object_proj.parameters():
            param.requires_grad = True
        for param in [self.class_embed]:
            param.requires_grad = True




        

    @classmethod
    def from_config(cls, cfg, in_channels, lang_encoder, mask_classification, extra):
        ret = {}

        ret["lang_encoder"] = lang_encoder
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification

        enc_cfg = cfg['MODEL']['ENCODER']
        dec_cfg = cfg['MODEL']['DECODER']
        dataset = cfg['DATASETS']['TRAIN'][0]

        ret["hidden_dim"] = dec_cfg['HIDDEN_DIM']
        ret["dim_proj"] = cfg['MODEL']['DIM_PROJ']
        ret["num_queries"] = dec_cfg['NUM_OBJECT_QUERIES']
        ret["contxt_len"] = cfg['MODEL']['TEXT']['CONTEXT_LENGTH']

        # Transformer parameters:
        ret["nheads"] = dec_cfg['NHEADS']
        ret["dim_feedforward"] = dec_cfg['DIM_FEEDFORWARD']

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert dec_cfg['DEC_LAYERS'] >= 1
        ret["dec_layers"] = dec_cfg['DEC_LAYERS'] - 1
        ret["pre_norm"] = dec_cfg['PRE_NORM']
        ret["enforce_input_project"] = dec_cfg['ENFORCE_INPUT_PROJ']
        ret["mask_dim"] = enc_cfg['MASK_DIM']

        ret["task_switch"] = extra['task_switch']
        ret['dataset'] = dataset
        ret['mask_head_enabled'] = cfg['MODEL']['MASK_HEAD_ENABLED']

        ret['object_feature_size'] = cfg['MODEL']['TEXT_DECODER']['OBJECT_FEATURE_SIZE']
        ret['vocab_size'] = cfg['MODEL']['TEXT_DECODER']['VOCAB_SIZE']
        ret['hidden_size'] = cfg['MODEL']['TEXT_DECODER']['HIDDEN_SIZE']
        ret['num_layers'] = cfg['MODEL']['TEXT_DECODER']['NUM_LAYERS']
        ret['attention_heads'] = cfg['MODEL']['TEXT_DECODER']['ATTENTION_HEADS']
        ret['feedforward_size'] = cfg['MODEL']['TEXT_DECODER']['FEEDFORWARD_SIZE']
        ret['mask_future_positions'] = cfg['MODEL']['TEXT_DECODER']['MASK_FUTURE_POSITIONS']
        ret['padding_idx'] = cfg['MODEL']['TEXT_DECODER']['PADDING_IDX']
        ret['decoder_type'] = cfg['MODEL']['TEXT_DECODER']['DECODER_TYPE']
        ret['use_act_checkpoint'] = cfg['MODEL']['TEXT_DECODER']['USE_ACT_CHECKPOINT']
        ret['max_steps'] = cfg['MODEL']['TEXT_DECODER']['MAX_STEPS']
        ret['beam_size'] = cfg['MODEL']['TEXT_DECODER']['BEAM_SIZE']


        return ret


    def calculate_iou(self,mask1, mask2):
        
        intersection = (mask1 & mask2).float().sum((1, 2))
        union = (mask1 | mask2).float().sum((1, 2))
        iou = intersection / union
        return iou


    def match_masks(self,targets, outputs):
        batch_size = len(targets)
        num_tokens = outputs.shape[1]
        pred_masks = outputs # [batch_size, num_tokens, 64, 64]

        matched_results = []

        for batch_idx in range(batch_size):
            target_masks = targets[batch_idx]['masks']  # [num_targets, 256, 256]

            target_masks_downsampled = F.interpolate(target_masks.unsqueeze(1).float(), size=(pred_masks[0].shape[1], pred_masks[0].shape[2]), mode='bilinear', align_corners=False).squeeze(1)  # [num_targets, 64, 64]

            matched_tokens = []

            for target_mask in target_masks_downsampled:
                target_mask_expanded = target_mask.unsqueeze(0).expand(num_tokens, -1, -1)  # [num_tokens, 64, 64]
                ious = self.calculate_iou(target_mask_expanded > 0, pred_masks[batch_idx] > 0)
                
                best_token_id = torch.argmax(ious)
                best_score = ious[best_token_id]
                matched_tokens.append((best_token_id.item(), best_score.item()))
            
            matched_results.append(matched_tokens)

        return matched_results


    def forward(self, x, mask_features, mask = None, target_queries = None, target_densecap = None, task='seg', extra={}):
                # x is a list of multi-scale feature
        torch.cuda.empty_cache()
        self._activate_stages()

        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []
        predictions_bbox = []
        predictions_densecap = []
        global_src = src[0].permute(1, 2, 0)
        global_image_feature = self.global_pooling(global_src).squeeze(-1)
        self_tgt_mask = None
        self_tgt_mask = self.self_attn_mask[:,:self.num_queries,:self.num_queries].repeat(output.shape[1]*self.num_heads, 1, 1)

        # prediction heads on learnable query features

        results = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0], task='seg',target_densecap=target_densecap,global_image_feature=global_image_feature)
        predictions_class.append(results["outputs_class"])
        predictions_mask.append(results["outputs_mask"])
        predictions_bbox.append(results["outputs_bbox"])
        predictions_densecap.append(results["outputs_densecap"])
        attn_mask = results["attn_mask"]

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output, avg_attn = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=self_tgt_mask,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )   
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )



            results = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels], layer_id=i, task='seg',target_densecap=target_densecap,global_image_feature=global_image_feature)
            attn_mask = results["attn_mask"]
            predictions_class.append(results["outputs_class"])
            predictions_mask.append(results["outputs_mask"])
            predictions_bbox.append(results["outputs_bbox"])

            predictions_densecap.append(results["outputs_densecap"])
        if self.training:



            out = { 'pred_logits': predictions_class[-1],
                    'pred_masks': predictions_mask[-1],
                    'pred_boxes': predictions_bbox[-1],
                    'pred_tokens': output,
                    'pred_dense_captions': predictions_densecap[-1],
                    'aux_outputs': self._set_aux_loss(
                        predictions_class if self.mask_classification else None, predictions_mask, predictions_bbox, predictions_densecap
                    )
                    }
            return out
        else:
            object_features = None
            norm_decoder_output = self.decoder_norm(output.transpose(0, 1)).clone()
            # global_image_feature = norm_decoder_output[:,self.num_queries-1:self.num_queries].squeeze(1)
            output_x = norm_decoder_output[:,:self.num_queries-1]

            for idx,o in enumerate(output_x):
                out_b = output_x[idx]
                global_tokens_b = global_image_feature[idx:idx+1]

                for id,token in enumerate(out_b):

                    if object_features is None:
                        #with global feature
                        object_features = self.object_proj(torch.cat((out_b[id:id+1],global_tokens_b),dim=0).unsqueeze(0))
                        #w/o global feature
                        # object_features = self.object_proj(out_b[id:id+1]).unsqueeze(0)

                    else:
                        #with global feature
                        object_features = torch.cat((object_features,self.object_proj(torch.cat((out_b[id:id+1],global_tokens_b),dim=0)).unsqueeze(0)), dim=0)
                        #w/o global feature
                        # object_features = torch.cat((object_features,self.object_proj(out_b[id:id+1]).unsqueeze(0)),dim=0)


            object_features = _ScaleGradient.apply(object_features, 1.0 / 3)
            if object_features is not None:
                text_decoder_output = self.text_decoder({'object_features': object_features})
                text_dense = []
                text_dense_b = []
                for prediction in text_decoder_output['predictions']:
                    # convert text tokens to words
                    
                    description = self.tokenizer.decode(prediction.tolist()[1:], skip_special_tokens=True)
                    description = description.replace(" - ", "-").replace(" 's","'s")
                    text_dense_b.append(description)
                    if len(text_dense_b) == self.num_queries-1:
                        text_dense.append(text_dense_b)
                        text_dense_b = []
            out = {
                    'pred_dense': text_dense,
                    'pred_logits': predictions_class[-1],
                    'pred_masks': predictions_mask[-1],
                    'pred_boxes': predictions_bbox[-1],
                    # 'pred_captions': predictions_caption[-1],
                }
            return out         


    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size, layer_id=-1, task='seg',target_densecap=None,global_image_feature=None):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_bbox = [None for i in range(len(decoder_output))]
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bicubic", align_corners=False, antialias=True)

        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        # NOTE: fill False for cls token (JY)
        attn_mask[:, self.num_queries:self.num_queries+1].fill_(False)

        # recompute class token output.
        norm_decoder_output = decoder_output / (decoder_output.norm(dim=-1, keepdim=True) + 1e-7)
        obj_token = norm_decoder_output[:,:self.num_queries-1]
        cls_token = norm_decoder_output[:,self.num_queries-1:self.num_queries]

        sim = (cls_token @ obj_token.transpose(1,2)).softmax(-1)[:,0,:,None] # TODO include class token.
        cls_token = (sim * decoder_output[:,:self.num_queries-1]).sum(dim=1, keepdim=True)
        decoder_output = torch.cat((decoder_output[:,:self.num_queries-1], cls_token), dim=1)

        # compute class, mask and bbox.
        class_embed = decoder_output @ self.class_embed
        # HACK do not compute similarity if mask is not on
        outputs_class = self.lang_encoder.compute_similarity(class_embed, fake=(((not self.task_switch['mask']) and self.training)))
        outputs_bbox = self.bbox_embed(decoder_output).sigmoid()
            # outputs_bbox = self.bbox_embed(decoder_output[:,:self.num_queries-1]).sigmoid()
        output_densecap = None
        if target_densecap:
            object_descriptions = []

            output_x = self.decoder_norm(output)[:self.num_queries-1,:,:].permute(1,0,2)

            matched_tokens = self.match_masks(target_densecap, outputs_mask[:,:self.num_queries-1,:])
            for v in target_densecap:
                object_descriptions = object_descriptions+v["object_descriptions"]
            #grit
            begin_token = self.task_begin_tokens["DenseCap"]
            text_decoder_inputs = self.get_target_text_tokens(object_descriptions, attn_mask, begin_token)
            object_features = None
            # global_image_feature = norm_decoder_output[:,self.num_queries-1:self.num_queries].squeeze(1)
            for idx,matched_token in enumerate(matched_tokens):
                out_b = output_x[idx]
                global_tokens_b = global_image_feature[idx:idx+1]
                for id,token in enumerate(matched_token):
                    if object_features is None:
                        #with global feature
                        object_features = torch.cat((out_b[token[0]:token[0]+1],global_tokens_b),dim=0).unsqueeze(0)
                    else:
                        #with global feature
                        object_features = torch.cat((object_features,torch.cat((out_b[token[0]:token[0]+1],global_tokens_b),dim=0).unsqueeze(0)), dim=0)

            object_features = self.object_proj(object_features)
            object_features = _ScaleGradient.apply(object_features, 1.0 / 3)
            
            text_decoder_inputs.update({'object_features': object_features})
            output_densecap = self.text_decoder(text_decoder_inputs)
        results = {
            "outputs_class": outputs_class,
            "outputs_mask": outputs_mask,
            "outputs_bbox": outputs_bbox,
            "attn_mask": attn_mask,

            "outputs_densecap": output_densecap,
        }
        return results

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, outputs_boxes, outputs_captions):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b, "pred_boxes": c, "pred_dense_captions": d}
                for a, b, c, d in zip(outputs_class[:-1], outputs_seg_masks[:-1], outputs_boxes[:-1], outputs_captions[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]


@register_decoder
def get_pix2cap_interface(cfg, in_channels, lang_encoder, mask_classification, extra):
    return Pix2Cap_Transformer_Decoder(cfg, in_channels, lang_encoder, mask_classification, extra)
