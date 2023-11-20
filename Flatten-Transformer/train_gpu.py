import os
import time
import argparse
import datetime
import numpy as np
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from util.samplers import RASampler
import torch.backends.cudnn as cudnn
from timm.models import create_model
import torch.distributed as dist
from timm.utils import NativeScaler, get_state_dict, ModelEma
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from config import get_config
from models import build_model
from datasets import build_dataset
from util.lr_scheduler import build_scheduler
from util.optimizer import build_optimizer
from estimate_model import Plot_ROC, Predictor
from util.losses import DistillationLoss
from util.logger import create_logger
import util.utils as utils
from util.engine import train_one_epoch, validate, throughput
from util.utils import load_checkpoint, save_checkpoint, save_checkpoint_new, get_grad_norm, auto_resume_helper, reduce_tensor, load_pretrained
import warnings
warnings.filterwarnings('ignore')

def parse_option():
    parser = argparse.ArgumentParser('FLatten Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, default='cfgs/flatten_pvt_v2_b0.yaml', help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    parser.add_argument('--model-ema', type=bool, default=False)
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # easy config modification
    parser.add_argument('--batch-size', type=int, default=8, help="batch size for single GPU")
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--data_root', type=str, default='/usr/local/Huangshuqi/ImageData/flower_data', help='path to dataset')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--input_size', type=int, default=224, help='input size of every image')
    parser.add_argument('--pin_mem', type=bool, default=True)
    parser.add_argument('--nb_classes', type=int, default=5, help='number classes of your dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp', type=bool, default=True, help='Activate amp if your gpu supports amp')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--pretrained', type=str, help='Finetune 384 initial checkpoint.', default='')
    parser.add_argument('--find-unused-params', action='store_true', default=False)

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                                     "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    parser.add_argument('--train-mode', action='store_true')
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    parser.set_defaults(train_mode=True)

    parser.add_argument('--ThreeAugment', action='store_true')  # 3augment

    parser.add_argument('--src', action='store_true')  # simple random crop

    # Distillation parameters  distilled
    parser.add_argument('--distilled', action='store_true', default=False, help='Perform distilled ')
    parser.add_argument('--teacher-model', default='regnety_200mf', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main():
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    args, config = parse_option()
    print(args)

    # rank = int(os.environ["RANK"])
    # world_size = int(os.environ['WORLD_SIZE'])
    # world_size = utils.get_world_size()
    # local_rank = int(os.environ['LOCAL_RANK'])
    # torch.cuda.set_device(local_rank)
    utils.init_distributed_mode(args)

    seed = config.SEED + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.enabled = True
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * utils.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * utils.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * utils.get_world_size() / 512.0

    print('*****************')
    print('linear_scaled_lr is ', linear_scaled_lr)
    print('linear_scaled_warmup_lr is ', linear_scaled_warmup_lr)
    print('linear_scaled_min_lr is ', linear_scaled_min_lr)
    print('*****************')

    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    # config.LOCAL_RANK = local_rank
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=utils.get_rank(), name=f"{config.MODEL.NAME}")
    if utils.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

        # print config
    logger.info(config.dump())

    dataset_train, dataset_val = build_dataset(args)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    args.num_workers = args.num_workers if 'linux' in sys.platform else 0

    data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, sampler=sampler_train, pin_memory=args.pin_mem,
                                   collate_fn=dataset_train.collate_fn, num_workers=args.num_workers, drop_last=True)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val, pin_memory=args.pin_mem,
                                 drop_last=False, collate_fn=dataset_train.collate_fn, num_workers=args.num_workers)

    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    logger.info(str(model))

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    optimizer = build_optimizer(config, model)

    model_without_ddp = model
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], broadcast_buffers=True,
                                                    find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module

    loss_scaler = NativeScaler()
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    total_epochs = config.TRAIN.EPOCHS + config.TRAIN.COOLDOWN_EPOCHS

    criterion = LabelSmoothingCrossEntropy()
    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = nn.CrossEntropyLoss()

    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(args.device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )


    max_accuracy = 0.0

    if args.pretrained != '':
        load_pretrained(args.pretrained, model_without_ddp, logger)

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model, logger)
        max_accuracy = max(max_accuracy, acc1)
        torch.cuda.empty_cache()
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, total_epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, logger,
                        model_ema, loss_scaler, total_epochs)
        acc1, acc5, loss = validate(config, data_loader_val, model, logger)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")

        # if utils.get_rank() == 0 and ((epoch + 1) % config.SAVE_FREQ == 0 or (epoch + 1) == (total_epochs)):
        #     save_checkpoint_new(config, epoch + 1, model_without_ddp, max(max_accuracy, acc1), optimizer, lr_scheduler,
        #                         logger)

        if acc1 >= max_accuracy:
            save_checkpoint_new(config, epoch + 1, model_without_ddp, max(max_accuracy, acc1), optimizer, lr_scheduler,
                                logger, name='max_acc')
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

    # plot ROC curve and confusion matrix
    print('*******************STARTING PREDICT*******************')
    Predictor(model_without_ddp, data_loader_val, f'{config.OUTPUT}/max_acc.pth', args.device)
    Plot_ROC(model_without_ddp, data_loader_val, f'{config.OUTPUT}/max_acc.pth', args.device)

if __name__ == '__main__':
    main()