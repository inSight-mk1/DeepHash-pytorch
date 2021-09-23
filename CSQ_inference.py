import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from utils.tools import *
from network import *

import torch
import torch.optim as optim
import time
import numpy as np
from scipy.linalg import hadamard  # direct import  hadamrd matrix from scipy
import random
import cv2
from PIL import Image
from tqdm import tqdm
from collections import Counter


torch.multiprocessing.set_sharing_strategy('file_system')


def init_model(weight_path, bit, device):
    net = ResNet(bit).to(device)
    net.load_state_dict(torch.load(weight_path))
    return net


def load_image(image_path):
    return Image.open(image_path).convert('RGB')


def load_images(images_list_path, mode='all'):
    with open(images_list_path, 'r') as f:
        images_list = f.readlines()

    images = []
    images_path = []
    for line in images_list:
        image_path = line.split(' ')[0]
        images_path.append(image_path)
        if mode == 'all':
            images.append(load_image(image_path))

    return images, images_path


def single_inference(image_read, model, device):
    transformer = image_transform(224, 224, 'inference')
    image = transformer(image_read)
    image = torch.unsqueeze(image, 0)

    image = image.to(device)

    u = model(image).data.cpu().sign()
    u = u.numpy().astype('int')

    return u


def multi_inference(images, model, device):
    us = []
    for image in tqdm(images):
        u = single_inference(image, model, device)
        us.append(u)
    return np.concatenate(us)


if __name__ == '__main__':
    device = torch.device("cuda:0")
    weight_path = 'C:\\Users\\admin\\dongwei\\workspace\\project\\DeepHash-pytorch\\weights\\0.8765\\imagenet-0.9007409339896844-model.pt'
    bit = 256
    model = init_model(weight_path, bit, device)

    db_code_path = 'database.npy'
    db_u = None
    database_list_path = "D:\\dataset\\consecutive_vehicles_v1\\database.txt"
    if not os.path.exists(db_code_path):
        images, images_path = load_images(database_list_path)
        db_u = multi_inference(images, model, device)
        np.save(db_code_path, db_u)
        print('Successfully save database hash code.')
    else:
        db_u = np.load(db_code_path)
        images, images_path = load_images(database_list_path, mode='path_only')
        print('Load database successfully.')

    db_cls = []
    for path in images_path:
        cls = int(path.split('images\\')[1][:5])
        db_cls.append(cls)

    # inference_image_path = "D:\\dataset\\consecutive_vehicles_v1\\images\\00538\\162495516486.30148.png"
    test_list_path = "D:\\dataset\\consecutive_vehicles_v1\\test.txt"
    _, test_images_path = load_images(test_list_path, mode='path_only')
    test_cls = []
    for path in test_images_path:
        cls = int(path.split('images\\')[1][:5])
        test_cls.append(cls)

    right_cnt = 0
    for i, image_path in enumerate(test_images_path):
        inference_image_path = image_path
        image_read = load_image(inference_image_path)
        u = single_inference(image_read, model, device)

        hamm = CalcHammingDist(u, db_u)

        # min_dist = np.min(hamm)
        # match_ids = np.where(hamm==min_dist)[1]
        # match_id = np.argmin(hamm)
        match_ids = np.argpartition(hamm.ravel(), 10)[:10]
        # print(match_id, hamm[0, match_id])
        # print(images_path[match_id])
        re_ids = []
        for id in match_ids:
            re_id = db_cls[id]  # Retrieved items class id
            re_ids.append(re_id)
        pred_id = Counter(re_ids).most_common(1)[0][0]
        if pred_id == test_cls[i]:  # right
            right_cnt += 1
        print(image_path, pred_id, test_cls[i])

    acc = right_cnt / len(test_images_path)
    print('Acc: %.4f' % acc)
