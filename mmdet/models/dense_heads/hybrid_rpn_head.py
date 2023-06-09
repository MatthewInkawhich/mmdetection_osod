"""This file contains code to build Hybrid OLN-RPN.

Reference:
    "Learning Open-World Object Proposals without Learning to Classify",
        Aug 2021. https://arxiv.org/abs/2108.06753
        Dahun Kim, Tsung-Yi Lin, Anelia Angelova, In So Kweon and Weicheng Kuo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init, ConvModule
from mmcv.ops import batched_nms
from mmcv.runner import force_fp32

from mmdet.core import (anchor_inside_flags, build_anchor_generator,
                        build_assigner, build_bbox_coder, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms, unmap)
from mmdet.core.bbox import bbox_overlaps
from ..builder import HEADS, build_loss
from .oln_rpn_head import OlnRPNHead


@HEADS.register_module()
class HybridRPNHead(OlnRPNHead):
    """OLN-RPN head.
    
    Learning localization instead of classification at the proposal stage is
    crucial as it avoids overfitting to the foreground by classification. For
    training the localization quality estimation branch, we randomly sample
    `num` anchors having an IoU larger than `neg_iou_thr` with the matched
    ground-truth boxes. It is recommended to use 'centerness' in this stage. For
    box regression, we replace the standard box-delta targets (xyhw) with
    distances from the location to four sides of the ground-truth box (lrtb). We
    choose to use one anchor per feature location as opposed to 3 in the standad
    RPN, because we observe its better generalization as each anchor can ingest
    more data.
    """

    def __init__(self, loss_objectness, objectness_type='Centerness', lambda_cls=0.0, **kwargs):
        super(HybridRPNHead, self).__init__(loss_objectness, **kwargs)
        self.lambda_cls = lambda_cls
        self.qofl = False
        if loss_objectness['type'] == "QualityOnlyFocalLoss":
            self.qofl = True


    def loss_single(self, cls_score, bbox_pred, objectness_score, anchors,
                    labels, label_weights, bbox_targets, bbox_weights, 
                    objectness_targets, objectness_weights, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            objectness_score (Tensor): Box objectness scorees for each anchor
                point has shape (N, num_anchors, H, W) 
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor with
                shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            objectness_targets (Tensor): Center regresion targets of each anchor
                with shape (N, num_total_anchors)
            objectness_weights (Tensor): Objectness weights of each anchro with 
                shape (N, num_total_anchors)
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)

        # objectness loss
        objectness_targets = objectness_targets.reshape(-1)
        objectness_weights = objectness_weights.reshape(-1)
        assert self.cls_out_channels == 1, (
            'cls_out_channels must be 1 for objectness learning.')
        objectness_score = objectness_score.permute(0, 2, 3, 1).reshape(-1)

        if self.qofl:
            loss_objectness = self.loss_objectness(
                objectness_score,  # We need raw prediction for QOFL loss computation
                objectness_targets, 
                objectness_weights, 
                avg_factor=num_total_samples)
        else:
            loss_objectness = self.loss_objectness(
                objectness_score.sigmoid(), 
                objectness_targets, 
                objectness_weights, 
                avg_factor=num_total_samples)
        return loss_cls, loss_bbox, loss_objectness


    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level
                label_weights_list (list[Tensor]): Label weights of each level
                bbox_targets_list (list[Tensor]): BBox targets of each level
                bbox_weights_list (list[Tensor]): BBox weights of each level
                num_total_pos (int): Number of positive samples in all images
                num_total_neg (int): Number of negative samples in all images
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        assign_result = self.assigner.assign(
            anchors, gt_bboxes, gt_bboxes_ignore,
            None if self.sampling else gt_labels)
        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        # Assign objectness gt and sample anchors
        objectness_assign_result = self.objectness_assigner.assign(
            anchors, gt_bboxes, gt_bboxes_ignore, None)
        objectness_sampling_result = self.objectness_sampler.sample(
            objectness_assign_result, anchors, gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes

            # Sanlity check: left, right, top, bottom distances must be greater
            # than 0.
            valid_targets = torch.min(pos_bbox_targets,-1)[0] > 0
            bbox_targets[pos_inds[valid_targets], :] = (
                pos_bbox_targets[valid_targets])
            bbox_weights[pos_inds[valid_targets], :] = 1.0

            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        objectness_targets = anchors.new_zeros(
            num_valid_anchors, dtype=torch.float)
        objectness_weights = anchors.new_zeros(
            num_valid_anchors, dtype=torch.float)
        objectness_pos_inds = objectness_sampling_result.pos_inds
        objectness_neg_inds = objectness_sampling_result.neg_inds
        objectness_pos_neg_inds = torch.cat(
            [objectness_pos_inds, objectness_neg_inds])

        if len(objectness_pos_inds) > 0:
            # Centerness as tartet -- Default
            if self.objectness_type == 'Centerness':
                pos_objectness_bbox_targets = self.bbox_coder.encode(
                    objectness_sampling_result.pos_bboxes, 
                    objectness_sampling_result.pos_gt_bboxes)
                valid_targets = torch.min(pos_objectness_bbox_targets,-1)[0] > 0
                pos_objectness_bbox_targets[valid_targets==False,:] = 0
                top_bottom = pos_objectness_bbox_targets[:,0:2]
                left_right = pos_objectness_bbox_targets[:,2:4]
                pos_objectness_targets = torch.sqrt(
                    (torch.min(top_bottom, -1)[0] / 
                        (torch.max(top_bottom, -1)[0] + 1e-12)) *
                    (torch.min(left_right, -1)[0] / 
                        (torch.max(left_right, -1)[0] + 1e-12)))
            elif self.objectness_type == 'BoxIoU':
                pos_objectness_targets = bbox_overlaps(
                    objectness_sampling_result.pos_bboxes,
                    objectness_sampling_result.pos_gt_bboxes,
                    is_aligned=True)
            else:
                raise ValueError(
                    'objectness_type must be either "Centerness" (Default) or '
                    '"BoxIoU".')

            objectness_targets[objectness_pos_inds] = pos_objectness_targets
            objectness_weights[objectness_pos_inds] = 1.0   

        if len(objectness_neg_inds) > 0: 
            objectness_targets[objectness_neg_inds] = 0.0
            objectness_weights[objectness_neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

            # objectness targets
            objectness_targets = unmap(
                objectness_targets, num_total_anchors, inside_flags)
            objectness_weights = unmap(
                objectness_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result,
                objectness_targets, objectness_weights, 
                objectness_pos_inds, objectness_neg_inds, objectness_pos_neg_inds,
                objectness_sampling_result)



    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           objectness_scores,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (num_anchors * 4, H, W).
            objectness_score_list (list[Tensor]): Box objectness scorees for
                each anchor point with shape (N, num_anchors, H, W)
            mlvl_anchors (list[Tensor]): Box reference for each scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            # <
            rpn_objectness_score = objectness_scores[idx]
            # >

            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            
            assert self.use_sigmoid_cls, 'use_sigmoid_cls must be True.'
            rpn_cls_score = rpn_cls_score.reshape(-1)
            rpn_cls_scores = rpn_cls_score.sigmoid()

            rpn_objectness_score = rpn_objectness_score.permute(
                1, 2, 0).reshape(-1)
            rpn_objectness_scores = rpn_objectness_score.sigmoid()
            
            #####################################################################3
            ### HYBRID
            
            # OLN: We use the predicted objectness score (i.e., localization quality)
            # as the final RPN score output.
            #scores = rpn_objectness_scores

            #print("\n\nRPN...")
            #print("self.lambda_cls:", self.lambda_cls)
            #print("rpn_objectness_scores:", rpn_objectness_scores, rpn_objectness_scores.shape, rpn_objectness_scores.min(), rpn_objectness_scores.max())
            #print("rpn_cls_scores:", rpn_cls_scores, rpn_cls_scores.shape, rpn_cls_scores.min(), rpn_cls_scores.max())

            # Hybrid: We use a linear interpolation of predicted cls scores and 
            # predicted objectness scores as the final RPN score prediction.
            scores = self.lambda_cls*rpn_cls_scores + (1-self.lambda_cls)*rpn_objectness_scores
            #print("scores:", scores, scores.shape, scores.min(), scores.max())
            #exit()


            #####################################################################3
        
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            anchors = mlvl_anchors[idx]
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:cfg.nms_pre]
                scores = ranked_scores[:cfg.nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(
                scores.new_full((scores.size(0), ), idx, dtype=torch.long))

        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bbox_preds)
        proposals = self.bbox_coder.decode(
            anchors, rpn_bbox_pred, max_shape=img_shape)
        ids = torch.cat(level_ids)

        if cfg.min_bbox_size > 0:
            w = proposals[:, 2] - proposals[:, 0]
            h = proposals[:, 3] - proposals[:, 1]
            valid_inds = torch.nonzero(
                (w >= cfg.min_bbox_size)
                & (h >= cfg.min_bbox_size),
                as_tuple=False).squeeze()
            if valid_inds.sum().item() != len(proposals):
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
                ids = ids[valid_inds]

        #nms_cfg = dict(type='nms', iou_threshold=cfg.nms_thr)
        nms_cfg = cfg.nms

        # No NMS:
        dets = torch.cat([proposals, scores.unsqueeze(1)], 1)
        
        return dets

        
