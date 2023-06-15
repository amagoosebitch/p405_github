from mmcv import Config
from mmseg.apis import set_random_seed
from mmseg.utils import get_device
import imageio
from mmseg.models import build_segmentor
from mmcv.runner import load_checkpoint
from mmseg.apis import inference_segmentor

import argparse
import sys


def init_segmentor(config, checkpoint=None, device='cuda:0'):
    """Initialize a segmentor from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str, optional) CPU/CUDA device option. Default 'cuda:0'.
            Use 'cpu' for loading model on CPU.
    Returns:
        nn.Module: The constructed segmentor.
    """
    config.model.pretrained = None
    config.model.train_cfg = None
    model = build_segmentor(config.model)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        model.CLASSES = ('road', 'notroad')
        model.PALETTE = [[255, 255, 255], [0, 0, 0]]
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    # model.eval()
    return model


def init_model():
    cfg = Config.fromfile('ccnet_r50-d8_512x1024_80k_cityscapes.py')
    cfg.norm_cfg = dict(type='BN', requires_grad=True)
    cfg.model.backbone.norm_cfg = cfg.norm_cfg
    cfg.model.decode_head.norm_cfg = cfg.norm_cfg
    cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
    cfg.model.decode_head.num_classes = 2
    cfg.model.auxiliary_head.num_classes = 2

    cfg.data.samples_per_gpu = 8
    cfg.data.workers_per_gpu = 8

    cfg.img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    cfg.crop_size = (256, 256)
    cfg.train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(1024, 1024), ratio_range=(0.5, 2.0)),
        dict(type='RandomCrop', crop_size=cfg.crop_size, cat_max_ratio=0.75),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(type='Normalize', **cfg.img_norm_cfg),
        dict(type='Pad', size=cfg.crop_size, pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg']),
    ]

    cfg.test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(1024, 1024),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **cfg.img_norm_cfg),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]

    cfg.data.train.type = cfg.dataset_type
    cfg.data.train.data_root = cfg.data_root
    cfg.data.train.img_dir = ''
    cfg.data.train.ann_dir = ''
    cfg.data.train.pipeline = cfg.train_pipeline
    cfg.data.train.split = 'splits/train.txt'

    cfg.data.val.type = cfg.dataset_type
    cfg.data.val.data_root = cfg.data_root
    cfg.data.val.img_dir = ''
    cfg.data.val.ann_dir = ''
    cfg.data.val.pipeline = cfg.test_pipeline
    cfg.data.val.split = 'splits/val.txt'

    cfg.data.test.type = cfg.dataset_type
    cfg.data.test.data_root = cfg.data_root
    cfg.data.test.img_dir = ''
    cfg.data.test.ann_dir = ''
    cfg.data.test.pipeline = cfg.test_pipeline
    cfg.data.test.split = 'splits/val.txt'

    cfg.load_from = 'best_model.pth'

    cfg.work_dir = './work_dirs/tutorial'

    cfg.runner.max_iters = 20000
    cfg.evaluation.interval = 1000
    cfg.evaluation.save_best = 'mIoU'
    cfg.checkpoint_config.interval = 1000
    cfg.log_config = dict(  # config to register logger hook
        interval=1000,  # Interval to print the log
        hooks=[],
        init_kwargs={'project': "MMSeg"},
    )

    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)
    cfg.device = get_device()



    config_path = 'ccnet_r50-d8_512x1024_80k_cityscapes.py'
    checkpoint_path = 'best_model.pth'

    # init model and load checkpoint
    model = init_segmentor(cfg, checkpoint_path)

    return model


def main():
    PATH_TO_FILE = setup_and_parse(sys.argv[1:]).filename
    PATH_TO_MASK = PATH_TO_FILE.split('.')[0] + '_mask' + '.jpg'
    model = init_model()
    img = imageio.imread(f'{PATH_TO_FILE}')
    result = inference_segmentor(model, img)
    imageio.imwrite(f'{PATH_TO_MASK}', result)


def setup_and_parse(inp):
    parser = argparse.ArgumentParser(description='Скрипт определения дорог по снимку')
    parser.add_argument(
        '--path',
        required=True,
        dest='filename',
        help='Укажите полный или относительный путь до файла с картинкой, для которой хотите предсказать дороги',
    )
    args = parser.parse_args(inp)
    return args


if __name__ == '__main__':
    main()
