import argparse
import random
import numpy as np
import torch
import datasets
import util.misc as utils
import cv2
import skimage
import skimage.transform
import re
from util import box_ops
from PIL import Image
from torch.utils.data import DataLoader
from datasets import build_dataset
from models import build_model
from pathlib import Path
from nltk.corpus import wordnet as wn


def noun2synset(noun):
    return wn.synset_from_pos_and_offset(noun[0], int(noun[1:])).name() if re.match(r'n[0-9]*',
                                                                                    noun) else "'{}'".format(noun)


def visualize_bbox(image_path=None, num_roles=None, noun_labels=None, pred_bbox=None, pred_bbox_conf=None,
                   output_dir=None):
    image = cv2.imread(image_path)
    image_name = image_path.replace('\\', '/')
    image_name = image_name.split('/')[-1].split('.')[0]

    h, w = image.shape[0], image.shape[1]
    red_color = (232, 126, 253)
    green_color = (130, 234, 198)
    blue_color = (227, 188, 134)
    orange_color = (98, 129, 240)
    brown_color = (79, 99, 216)
    purple_color = (197, 152, 173)
    colors = [red_color, green_color, blue_color, orange_color, brown_color, purple_color]
    white_color = (255, 255, 255)
    line_width = 2
    res_color = {}
    # the value of pred_bbox_conf is logit, not probability.
    for i in range(num_roles):
        if pred_bbox_conf[i] >= 0:
            # bbox
            pred_left_top = (int(pred_bbox[i][0].item()), int(pred_bbox[i][1].item()))
            pred_right_bottom = (int(pred_bbox[i][2].item()), int(pred_bbox[i][3].item()))
            lt_0 = max(pred_left_top[0], line_width)
            lt_1 = max(pred_left_top[1], line_width)
            rb_0 = min(pred_right_bottom[0], w - line_width)
            rb_1 = min(pred_right_bottom[1], h - line_width)
            lt = (lt_0, lt_1)
            rb = (rb_0, rb_1)
            cv2.rectangle(img=image, pt1=lt, pt2=rb, color=colors[i], thickness=line_width, lineType=-1)
            label = noun_labels[i].split('.')[0]
            # label = noun_labels[i]
            res_color[label] = colors[i]
            print(res_color)

            text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            p1 = (lt[0], lt[1] - text_size[1])
            cv2.rectangle(img=image, pt1=(p1[0], (p1[1] - 2 - baseline)),
                          pt2=((p1[0] + text_size[0]), (p1[1] + text_size[1])), color=colors[i], thickness=-1)
            cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, white_color, 1, 8)

            # save image
    cv2.imwrite("{}/{}_result.jpg".format(output_dir, image_name), image)
    print('-' * 20 + " output bbox success " + '-' * 20)

    return res_color


def process_image(image):
    mean = np.array([[[0.485, 0.456, 0.406]]])
    std = np.array([[[0.229, 0.224, 0.225]]])
    image = (image.astype(np.float32) - mean) / std
    min_side, max_side = 300, 400
    rows_orig, cols_orig, cns_orig = image.shape
    smallest_side = min(rows_orig, cols_orig)
    scale = min_side / smallest_side
    largest_side = max(rows_orig, cols_orig)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    # resize the image with the computed scale
    image = skimage.transform.resize(image, (int(round(rows_orig * scale)), int(round((cols_orig * scale)))))
    rows, cols, cns = image.shape
    new_image = np.zeros((rows, cols, cns)).astype(np.float32)
    new_image[:rows, :cols, :] = image.astype(np.float32)
    image = torch.from_numpy(new_image)

    shift_1 = int((400 - cols) * 0.5)
    shift_0 = int((400 - rows) * 0.5)

    max_height = 400
    max_width = 400
    padded_imgs = torch.zeros(1, max_height, max_width, 3)
    padded_imgs[0, shift_0:shift_0 + image.shape[0], shift_1:shift_1 + image.shape[1], :] = image
    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    height = torch.tensor(int(image.shape[0])).float()
    width = torch.tensor(int(image.shape[1])).float()
    shift_0 = torch.tensor(shift_0).float()
    shift_1 = torch.tensor(shift_1).float()
    scale = torch.tensor(scale).float()
    mw = torch.tensor(max_width).float()
    mh = torch.tensor(max_height).float()

    return (utils.nested_tensor_from_tensor_list(padded_imgs),
            {'width': width,
             'height': height,
             'shift_0': shift_0,
             'shift_1': shift_1,
             'scale': scale,
             'max_width': mw,
             'max_height': mh}
            )


def predict(model, device, image_path=None, inference=False,
            idx_to_verb=None, idx_to_role=None,
            vidx_ridx=None, idx_to_class=None, output_dir=None):
    model.eval()
    image_name = image_path.replace('\\', '/')
    image_name = image_name.split('/')[-1].split('.')[0]

    # load image & process
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = np.array(image)
    image = image.astype(np.float32) / 255.0
    image, info = process_image(image)
    image = image.to(device)
    info = {k: v.to(device) if type(v) is not str else v for k, v in info.items()}
    # print(info)

    output = model(image, inference=inference)
    pred_verb = output['pred_verb'][0]
    pred_noun = output['pred_noun'][0]
    pred_bbox = output['pred_bbox'][0]
    pred_bbox_conf = output['pred_bbox_conf'][0]

    top1_verb = torch.topk(pred_verb, k=1, dim=0)[1].item()
    roles = vidx_ridx[top1_verb]
    num_roles = len(roles)
    verb_label = idx_to_verb[top1_verb]
    role_labels = []
    noun_labels = []
    print('-' * 20 + "model success, ready predict boxes" + '-' * 20)
    for i in range(num_roles):
        top1_noun = torch.topk(pred_noun[i], k=1, dim=0)[1].item()
        role_labels.append(idx_to_role[roles[i]])
        noun_labels.append(noun2synset(idx_to_class[top1_noun]))

    # convert bbox
    mw, mh = info['max_width'], info['max_height']
    w, h = info['width'], info['height']
    shift_0, shift_1, scale = info['shift_0'], info['shift_1'], info['scale']
    pb_xyxy = box_ops.swig_box_cxcywh_to_xyxy(pred_bbox.clone(), mw, mh, device=device)
    for i in range(num_roles):
        pb_xyxy[i][0] = max(pb_xyxy[i][0] - shift_1, 0)
        pb_xyxy[i][1] = max(pb_xyxy[i][1] - shift_0, 0)
        pb_xyxy[i][2] = max(pb_xyxy[i][2] - shift_1, 0)
        pb_xyxy[i][3] = max(pb_xyxy[i][3] - shift_0, 0)
        # locate predicted boxes within image (processing w/ image width & height)
        pb_xyxy[i][0] = min(pb_xyxy[i][0], w)
        pb_xyxy[i][1] = min(pb_xyxy[i][1], h)
        pb_xyxy[i][2] = min(pb_xyxy[i][2], w)
        pb_xyxy[i][3] = min(pb_xyxy[i][3], h)
    pb_xyxy /= scale

    print('-' * 20 + " predict boxes success , ready output txt" + '-' * 20)
    # outputs
    with open("{}/{}_result.txt".format(output_dir, image_name), "w") as f:
        text_line = "verb: {} \n".format(verb_label)
        f.write(text_line)
        for i in range(num_roles):
            text_line = "role: {}, noun: {} \n".format(role_labels[i], noun_labels[i])
            f.write(text_line)
        f.close()

    print('-' * 20 + " output txt success, ready output bbox " + '-' * 20)
    return visualize_bbox(image_path=image_path, num_roles=num_roles, noun_labels=noun_labels, pred_bbox=pb_xyxy,
                   pred_bbox_conf=pred_bbox_conf, output_dir=output_dir)
