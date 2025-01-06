# ------------------------------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/criterion.py
# Modified by Zuyao You (https://github.com/geshang777)
# ------------------------------------------------------------------------------------------------------
"""
MaskFormer criterion.
"""
import logging

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from timm.loss import SoftTargetCrossEntropy
from .point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..language.loss import ql_multi_contrastive_loss, image_text_contrastive_loss_queue, vl_similarity, all_gather_grad
from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list, _max_by_axis
from ..utils import box_ops

# from image2html.visualizer import VL
logger = logging.getLogger(__name__)

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, top_x_layers, losses,
                 num_points, oversample_ratio, importance_sample_ratio):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.top_x_layers = top_x_layers
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(self, outputs, targets, indices, num_masks, layer_id, extra):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        if layer_id > self.top_x_layers['mask']:
            return {"loss_mask_ce_0": 0}

        if indices is None or len(targets) == 0:
            loss_ce = outputs['pred_logits'].sum() * 0.0
            losses = {"loss_mask_ce_0": loss_ce}
            return losses

        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].type(self.empty_weight.dtype)

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        if src_logits.shape[2] == self.num_classes+1:
            empty_weight = torch.ones(self.num_classes + 1).to(src_logits.device).type(self.empty_weight.dtype)
            empty_weight[-1] = self.eos_coef
        else:
            empty_weight = torch.ones(self.num_classes + 1000 + 1).to(src_logits.device).type(self.empty_weight.dtype)
            empty_weight[self.num_classes] = self.eos_coef

        # print(labels,src_logits.shape)#torch.Size([1, 100, 134])
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, empty_weight)
        losses = {"loss_mask_ce_0": loss_ce}
        return losses


    def loss_mask_captionings(self, outputs, targets, indices, num_masks, layer_id, extra):
        if layer_id >= self.top_x_layers['densecaption']:
            return {"loss_densecaptioning_0": 0}
        log_soft = nn.LogSoftmax(dim=1)
        kl = nn.KLDivLoss(reduction='none')
        feature, target = outputs["pred_dense_captions"]
        feature = feature.float().to(torch.float32)
        valid_mask = target != 0
        target = target[valid_mask]
        feature = feature[valid_mask]
        assert target.numel() > 0
        eps = 0.1
        n_class = feature.size(1)
        one_hot = torch.zeros_like(feature).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = log_soft(feature)
        loss_caption = kl(log_prb, one_hot).sum(dim=1).mean()

        # loss_caption = outputs["pred_dense_captions"]#.to(torch.float32)
        losses = {"loss_densecaption_0": loss_caption}
        return losses


    def loss_masks(self, outputs, targets, indices, num_masks, layer_id, extra):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        if layer_id >= self.top_x_layers['mask']:
            return {"loss_mask_bce_0": 0, "loss_mask_dice_0": 0}

        assert "pred_masks" in outputs
        if indices is None or len(targets) == 0:
            loss = outputs['pred_masks'].sum() * 0.0
            losses = {"loss_mask_bce_0": loss, "loss_mask_dice_0": loss}
            return losses
        

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        # import pdb;pdb.set_trace()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]
        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]
        # import  pdb; pdb.set_trace()
        
        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            ).type(src_masks.dtype)
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask_bce_0": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_mask_dice_0": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

  
    def loss_boxes(self, outputs, targets, indices, num_boxes, layer_id, extra):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        if layer_id >= self.top_x_layers['bbox']:
            return {"loss_bbox_box_0": 0, "loss_bbox_giou_0": 0}

        assert 'pred_boxes' in outputs

        if indices is None or len(targets) == 0:
            loss = outputs['pred_boxes'].sum() * 0.0
            losses = {"loss_bbox_0": loss, "loss_giou_0": loss}
            return losses

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"]
        src_boxes = src_boxes[src_idx]
        
        target_boxes = [t['boxes'] for t in targets]
        max_size = _max_by_axis([list(box.shape) for box in target_boxes])
        max_size = [len(target_boxes)] + max_size
        empty_boxes = torch.zeros(max_size).to(src_boxes.device)
        for idx, tar_box in enumerate(target_boxes):
            empty_boxes[idx,:tar_box.shape[0],:] = tar_box
        target_boxes = empty_boxes[tgt_idx]

        # target_isthings = [t['is_things'] for t in targets]
        # max_size = _max_by_axis([list(lab.shape) for lab in target_isthings])
        # max_size = [len(target_isthings)] + max_size
        # empty_lab = torch.zeros(max_size).to(src_boxes.device)

        # for idx, tar_thing in enumerate(target_isthings):
        #     empty_lab[idx,:tar_thing.shape[0]] = tar_thing
        # target_isthings = empty_lab[tgt_idx]

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox_box_0'] = loss_bbox.sum() / num_boxes
        
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_bbox_giou_0'] = loss_giou.sum() / num_boxes
        return losses
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks, layer_id, extra):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
            'bbox': self.loss_boxes,

            'densecaptions': self.loss_mask_captionings,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks, layer_id, extra)

    def forward(self, outputs, targets, extra=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets) #, mode='caption_wbbox')

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs_without_aux.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks, 0, extra))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            # NOTE: we reverse the aux_outputs so that the first is the second last layer
            for i, aux_outputs in enumerate(outputs["aux_outputs"][::-1]):
                indices = self.matcher(aux_outputs, targets) #, mode='caption_wbbox')
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks, (i+1), extra)
                    l_dict = {k.replace('_0', f"_{i+1}"): v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
