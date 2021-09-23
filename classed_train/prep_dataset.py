import os
import random
import shutil
from tqdm import tqdm

src_path = 'D:\\dataset\\consecutive_vehicles_v1\\labeled'
dst_path = 'D:\\dataset\\consecutive_vehicles_v1\\dataset'

train_path = os.path.join(dst_path, 'train')
val_path = os.path.join(dst_path, 'val')
test_path = os.path.join(dst_path, 'test')

train_ratio = 0.9
test_ratio = 0.0
val_ratio = 0.1


if not os.path.exists(train_path):
    os.mkdir(train_path)

if not os.path.exists(test_path):
    os.mkdir(test_path)

if not os.path.exists(val_path):
    os.mkdir(val_path)

cls_paths = os.listdir(src_path)
for cls in cls_paths:
    cls_path = os.path.join(src_path, cls)
    cls_files = os.listdir(cls_path)
    total_cnt = len(cls_files)
    train_cnt = int(total_cnt * train_ratio)
    test_cnt = int(total_cnt * test_ratio)
    val_cnt = total_cnt - train_cnt - test_cnt
    train_files = random.sample(cls_files, train_cnt)
    left_files = list(set(cls_files) - set(train_files))
    val_files = random.sample(left_files, val_cnt)
    test_files = list(set(left_files) - set(val_files))
    if len(test_files) > test_cnt:
        test_files = random.sample(test_files, test_cnt)

    dst_train_path = os.path.join(train_path, cls)
    os.mkdir(dst_train_path)
    for f in tqdm(train_files):
        simg_path = os.path.join(cls_path, f)
        dimg_path = os.path.join(dst_train_path, f)
        shutil.copyfile(simg_path, dimg_path)

    dst_test_path = os.path.join(test_path, cls)
    os.mkdir(dst_test_path)
    for f in tqdm(test_files):
        simg_path = os.path.join(cls_path, f)
        dimg_path = os.path.join(dst_test_path, f)
        shutil.copyfile(simg_path, dimg_path)

    dst_val_path = os.path.join(val_path, cls)
    os.mkdir(dst_val_path)
    for f in tqdm(val_files):
        simg_path = os.path.join(cls_path, f)
        dimg_path = os.path.join(dst_val_path, f)
        shutil.copyfile(simg_path, dimg_path)
