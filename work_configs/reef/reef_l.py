#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 1
        self.depth = 1.0
        self.width = 1.0
        self.input_size = (800, 1280)
        self.test_size = (800, 1280)

        self.warmup_epochs = 1
        self.max_epoch = 20
        self.basic_lr_per_img = 0.01 / 64.0
        self.no_aug_epochs = 1

        holdout = 1
        self.data_dir = "./data/reef/train"
        self.train_ann = f"annotations/fold_holdout{holdout}.json"
        self.train_image_dir = "images"
        self.val_ann = f"annotations/fold_{holdout}.json"
        self.val_image_dir = "images"
        self.test_ann = f"annotations/fold_{holdout}.json"
        self.test_image_dir = "images"
        self.filter_empty_gt = True

        self.output_dir = "./work_dirs"
        self.exp_name = "ylx_l"
        self.print_interval = 50
        self.eval_interval = 1
