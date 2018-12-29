# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import errno
import os
import socket
from datetime import datetime


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def get_run_name():
    """ A unique name for each run, specific to minute"""
    # return datetime.now().strftime('%b%d-%H-%M-%S') + '_' + socket.gethostname()
    return datetime.now().strftime('%b%d-%H-%M') + '_' + socket.gethostname()


def get_output_dir(args, run_name, output_dir):
    """ Get root output directory for each run """
    cfg_filename, _ = os.path.splitext(os.path.split(args.config_file)[1])
    return os.path.join(output_dir, cfg_filename, run_name)

