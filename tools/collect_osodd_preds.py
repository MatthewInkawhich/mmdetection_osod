# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings
import random
import json
import numpy as np

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import (multi_gpu_test, single_gpu_test, single_gpu_osodd_test,
                        multi_gpu_osodd_test)
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import (build_ddp, build_dp, compat_cfg, get_device,
                         replace_cfg_vals, rfnext_init_model,
                         setup_multi_processes, update_data_root)


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('datasplit', help='train|val|test')

    ## OSODD
    parser.add_argument(
        '--num-images',
        type=int,
        default=0,
        help='Number of images to consider (leave at 0 for all)')
    parser.add_argument(
        '--score-thresh',
        type=float,
        default=0.0,
        help='Score threshold for keeping a pred')
    parser.add_argument(
        '--lite',
        action='store_true',
        help='whether to store penultimate_feats.')


    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()
    args.datasplit = args.datasplit.lower()
    assert (args.datasplit == 'train' or args.datasplit == 'val' or args.datasplit == 'test'), \
        f"Error: args.datasplit: {args.datasplit} is not valid"    
    print("args.datasplit:", args.datasplit)
    ##########################################################
    ### SETUP
    ##########################################################
    ### Prepare configs
    cfg = Config.fromfile(args.config)
    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)
    # update data root according to MMDET_DATASETS
    update_data_root(cfg)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    elif 'init_cfg' in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    cfg.gpu_ids = [args.gpu_id]
    cfg.device = get_device()
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # Update score thresh
    cfg.model.test_cfg.rcnn.score_thr = args.score_thresh
    print("cfg.model.test_cfg.rcnn.score_thr:", cfg.model.test_cfg.rcnn.score_thr)

    # Update data according to args.datasplit
    if args.datasplit == "train":
        cfg.data.test.classes = cfg.data.train.classes
        cfg.data.test.ann_file = cfg.data.train.ann_file
        cfg.data.test.img_prefix = cfg.data.train.img_prefix
    elif args.datasplit == "val":
        cfg.data.test.classes = cfg.data.osodd_val.classes
        cfg.data.test.ann_file = cfg.data.osodd_val.ann_file
        cfg.data.test.img_prefix = cfg.data.osodd_val.img_prefix
    else:
        cfg.data.test.classes = cfg.data.osodd_test.classes
        cfg.data.test.ann_file = cfg.data.osodd_test.ann_file
        cfg.data.test.img_prefix = cfg.data.osodd_test.img_prefix


    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }



    ##########################################################
    ### BUILD DATALOADER AND DETECTOR
    ##########################################################
    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    if rank == 0:
        print("\nmodel:", model)

    # init rfnext if 'RFSearchHook' is defined in cfg
    rfnext_init_model(model, cfg=cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is None and cfg.get('device', None) == 'npu':
        fp16_cfg = dict(loss_scale='dynamic')
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES


    ##########################################################
    ### GENERATE PREDICTIONS ON TEST IMAGES
    ##########################################################
    # Optionally subsample approx args.num_images images
    # We do this using probabilistic sampling
    image_prob = 100.0
    if args.num_images:
        if args.num_images < len(dataset):
            image_prob = args.num_images / len(dataset)
            print("image_prob:", image_prob)

    if not distributed:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
        outputs = single_gpu_osodd_test(model, data_loader, image_prob)
    else:
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False)

        # In multi_gpu_test, if tmpdir is None, some tesnors
        # will init on cuda by default, and no device choice supported.
        # Init a tmpdir to avoid error on npu here.
        if cfg.device == 'npu' and args.tmpdir is None:
            args.tmpdir = './npu_tmpdir'

        outputs = multi_gpu_osodd_test(
            model, data_loader, args.tmpdir, args.gpu_collect
            or cfg.evaluation.get('gpu_collect', False), image_prob)


    ##########################################################
    ### FORMAT AND SAVE PREDS
    ##########################################################
    rank, _ = get_dist_info()
    if rank == 0:
#        subsample = False
#        # Optional arg specifies how many images to consider
#        random.seed(0)
#        if args.num_images:
#            if args.num_images < len(outputs):
#                print("\n\nSampling {} images".format(args.num_images))
#                subsample = True
#                all_indices = list(range(len(outputs)))
#                indices_to_consider = set(random.sample(all_indices, args.num_images))
#                #new_outputs = []
#                for i in range(len(outputs)):
#                    if i not in indices_to_consider:
#                        outputs[i] = []
#                    #if i in indices_to_consider:
#                    #    new_outputs.append(outputs[i])
#                    #else:
#                    #    new_outputs.append([])
#                #outputs = new_outputs

        print("\n\nlen(outputs):", len(outputs))
        full_count = 0
        for i in range(len(outputs)):
            if outputs[i]:
                full_count += 1
        print("full_count:", full_count)


        #for i in range(5):
        #    print("\n", i, outputs[i], len(outputs[i]))
        if not args.lite:
            results, logits, penultimate_feats = dataset._det2json_osodd(outputs, save_penultimate_feat=True)
        else:
            results, logits = dataset._det2json_osodd(outputs, save_penultimate_feat=False)

        print("\n\nresults:", len(results))
        logits_np = np.vstack(logits)
        print("logits:", logits_np.shape, logits_np.dtype)
        if not args.lite:
            penultimate_feats_np = np.vstack(penultimate_feats)
            print("penultimate_feats:", penultimate_feats_np.shape, penultimate_feats_np.dtype)


        # Write all predicted instances in COCO-format to a json file 
        # Used by a COCO evaluation later.
        score_thr = "{:.2f}".format(args.score_thresh).split('.')[-1]
        suffix = '_' + args.datasplit + '_t' + score_thr
        if image_prob < 100.0:
            suffix += '_' + str(args.num_images) + 'imgs'

        
        osodd_output_path = os.path.join(os.path.dirname(args.checkpoint), 'osodd_predictions' + suffix + '.json')
        print("\nSaving OSODD predictions to:", osodd_output_path, "...")
        with open(osodd_output_path, 'w') as fp:
            json.dump(results, fp, indent=4, separators=(',', ': '))
        
        logits_output_path = os.path.join(os.path.dirname(args.checkpoint), 'osodd_logits' + suffix + '.npy')
        print("\nSaving OSODD logits to:", logits_output_path, "...")
        with open(logits_output_path, 'wb') as fp:
            np.save(fp, logits_np)

        if not args.lite:
            penultimate_feats_output_path = os.path.join(os.path.dirname(args.checkpoint), 'osodd_penultimate_feats' + suffix + '.npy')
            print("\nSaving OSODD penultimate_feats to:", penultimate_feats_output_path, "...")
            with open(penultimate_feats_output_path, 'wb') as fp:
                np.save(fp, penultimate_feats_np)

        
        




if __name__ == '__main__':
    main()
