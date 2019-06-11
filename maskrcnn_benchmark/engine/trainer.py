# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.utils.metric_logger import TensorboardLogger


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k, v in loss_dict.items():
            loss_names.append(k)
            all_losses.append(v)
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    snapshot,
    tb_log_dir,
    tb_exp_name,
    use_tensorboard=False
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = TensorboardLogger(log_dir=tb_log_dir,
                               exp_name=tb_exp_name,
                               start_iter=arguments['iteration'],
                               delimiter="  ") \
        if use_tensorboard else MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    for iteration, (images, targets, img_idx) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        # we only sum up the loss but no other metrics
        losses = sum([v for (k, v) in loss_dict.items() if k.split('/')[-1][:4] == 'loss'])
        losses.backward()
        optimizer.step()

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum([v for (k, v) in loss_dict_reduced.items() if k.split('/')[-1][:4]=='loss'])

        # Prepend the key name so as to allow better visualisation in tensorboard
        loss_dict_reduced['detector_losses/total_loss'] = losses_reduced
        meters.update(**loss_dict_reduced)
        meters.update(lr=optimizer.param_groups[-1]['lr'])

        batch_time = time.time() - end
        end = time.time()
        process_time = {'time/batch_time': batch_time, 'time/data_time': data_time}
        meters.update(**process_time)

        eta_seconds = meters.meters['time/batch_time'].global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % snapshot == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "\n \t eta: {eta}",
                        "iter: {iter}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                        "\n \t {meters}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    meters=str(meters),
                )
            )
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / (max_iter)))
