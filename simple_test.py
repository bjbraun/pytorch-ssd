import argparse
import os
import logging
import sys
import itertools

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from vision.ssd.ssd import MatchPrior
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.datasets.own_dataset import VOCDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform

import tensorflow as tf
import datetime

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')

parser.add_argument('--datasets', nargs='+', help='Dataset directory path')
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
args = parser.parse_args()

if __name__ == '__main__':
    timer = Timer()
    config = mobilenetv1_ssd_config
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)
    logging.info(args)
    for dataset_path in args.datasets:
        #dataset = VOCDataset(dataset_path, transform=train_transform,
        #                         target_transform=target_transform)
        dataset = VOCDataset(dataset_path, is_test=True)

    #VOCDataset

    # __init__
    #print(dataset.root) #Pfad zu Voc_own
    #print(dataset.ids) # Sind alle Zahlen aus dem File Imagesets/Main/trainval.txt
    #print(dataset.class_dict) # Sind alle Klassen mit Zahlen durchnummeriert (z.B. 1 = aeroplane)
    #print(dataset.filenames_img[0])
    #print(dataset.filenames_bbox[0])


    # __getitem__ (kann einfach aufgerufen werden mit objekt[i], objekt von VOCDataset)
    # _get_annotation gibt für eine image_id aus ids die bndbox und das label für das jeweilige image zurück nachdem es aus dem xml file ausgelesen wurde
    # __get_item__ gibt dann die bndbox, label und image zurück von einer image_id
    i,b,labels = dataset[400]

    writer = tf.summary.create_file_writer("./logs")

    with writer.as_default():
        for step in range(10):
            tf.summary.scalar("loss", 10-step, step=step)
            writer.flush()

    #print(l)
    #id,bld = dataset.get_annotation(0)
    #print(id)
    #print(bld[0])
    #print(id)
    #print(bld[0][1]) # Gibt die zweite Reihe von den Bounding Boxes wieder

    #i = dataset.get_image(1) # ACHTUNG: Kann man so nicht callen, da man boxes, etc noch braucht. Aber passt so

