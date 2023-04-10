import argparse
import random
import numpy as np
import torch
from flask import g

# import datasets
import util.misc as utils
import cv2
import skimage
import skimage.transform
import nltk
import re
from util import box_ops
from PIL import Image
from torch.utils.data import DataLoader
from datasets import build_dataset
from models import build_model
from pathlib import Path
from nltk.corpus import wordnet as wn


def get_args_parser():
    parser = argparse.ArgumentParser('Set grounded situation recognition transformer', add_help=False)
    # parser.add_argument('', default='', type=str)
    # Backbone parameters
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--position_embedding', default='learned', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # Transformer parameters
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.15, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")

    # Dataset parameters
    parser.add_argument('--dataset_file', default='swig')
    parser.add_argument('--swig_path', type=str, default="SWiG")
    parser.add_argument('--image_path', default='static/input/image.jpg',
                        help='path where the test image is')

    # Etc...
    parser.add_argument('--inference', default=True)
    parser.add_argument('--output_dir', default='static/output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for init')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--saved_model', default='gsrtr_checkpoint.pth',
                        help='path where saved model is')

    return parser


class GSRTRansfomer(object):
    def __init__(self):
        nltk.download('wordnet')
        parser = argparse.ArgumentParser('GSRTR init script', parents=[get_args_parser()])
        self.args = parser.parse_args()
        if self.args.output_dir:
            Path(self.args.output_dir).mkdir(parents=True, exist_ok=True)
        self.init_model(self.args)
        g.args = self.args

    def init_model(self, args):
        # fix the seed
        seed = args.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # num noun classes in train dataset
        dataset_train = build_dataset(image_set='train', args=args)
        args.num_noun_classes = dataset_train.num_nouns()

        # build model
        print('-' * 20+"build model"+'-' * 20)
        print(args.device)
        device = torch.device(args.device)
        model, _ = build_model(args)
        model.to(device)
        args.device = device
        print(args.device)
        checkpoint = torch.load(args.saved_model, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        args.model = model
        print('-' * 20+"build model success"+'-' * 20)

