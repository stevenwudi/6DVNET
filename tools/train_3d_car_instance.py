"""
Training script for
Apolloscape car instance challenge
http://apolloscape.auto/car_instance.html
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import matplotlib
matplotlib.use("TkAgg")
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir, get_run_name, get_output_dir


def parse_args():
    """Parse input arguments"""

    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument("--config-file", default="../configs/e2e_3d_car_101_FPN_triple_head.yaml", metavar="FILE", help="path to config file", type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--skip-test", default=False, dest="skip_test", help="Do not test the final model")

    # Optional
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--dataset', dest='dataset', default='ApolloScape', help='Dataset to use')
    parser.add_argument('--disp_interval', help='Display training info every N iterations', default=20, type=int)
    parser.add_argument('--no_cuda', dest='cuda', help='Do not use CUDA device', action='store_false')
    parser.add_argument('--output_dir', default='/media/SSD_1TB/ApolloScape/6DVNET_experiments')

    # Epoch
    parser.add_argument('--start_step', help='Starting step count for training epoch. 0-indexed.', default=0, type=int)
    # Resume training: requires same iterations per epoch
    parser.add_argument('--resume', default=False, help='resume to training on a checkpoint', action='store_true')
    parser.add_argument('--no_save', action='store_true', help='do not save anything')
    parser.add_argument('--load_ckpt', default='.', help='checkpoint path to load', type=str)

    parser.add_argument('--ckpt_ignore_head', default=[], help='heads parameters will be ignored during loading')
    parser.add_argument('--use-tensorboard', default=True, help='Use tensorflow tensorboard to log training info', action='store_true')

    return parser.parse_args()


def train(cfg, local_rank, distributed, use_tensorboard=False, logger=None):
    arguments = {"iteration": 0}
    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(cfg, model, optimizer, scheduler, output_dir, save_to_disk, logger=logger)
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    tensorboard_logdir = cfg.OUTPUT_DIR
    tensorboard_exp_name = cfg.TENSORBOARD_EXP_NAME
    snapshot = cfg.SOLVER.SNAPSHOT_ITERS

    do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        snapshot,
        tensorboard_logdir,
        tensorboard_exp_name,
        use_tensorboard=use_tensorboard
    )

    return model


def main():
    args = parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    ### Training Setups ###
    args.run_name = get_run_name() + '_step'
    output_dir = get_output_dir(args, args.run_name, args.output_dir)
    args.cfg_filename = os.path.basename(args.config_file)
    cfg.OUTPUT_DIR = output_dir
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = train(
        cfg=cfg,
        local_rank=args.local_rank,
        distributed=args.distributed,
        use_tensorboard=args.use_tensorboard,
        logger=logger,
    )


if __name__ == "__main__":
    main()
