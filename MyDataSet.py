# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/27 20:51
# @Author  : Jis-Baos
# @File    : MyDataSet.py

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class MyDataSet(Dataset):
    def __init__(self, dataset_dir, mode="train", trans=None):
        self.data_dir = dataset_dir
        self.image_dir = os.path.join(dataset_dir, "Images")
        self.label_dir = os.path.join(dataset_dir, "Labels_encoder")
        self.imagesets_dir = os.path.join(dataset_dir, "ImageSets")
        self.mode = mode

        # img_list存的只是图片的名字
        self.img_list = []
        if mode == "train":
            img_list_txt = os.path.join(self.imagesets_dir, mode + '.txt')
            with open(img_list_txt, 'r') as f:
                for line in f.readlines():
                    self.img_list.append(line.strip('\n'))

        elif mode == "val":
            img_list_txt = os.path.join(self.imagesets_dir, mode + '.txt')
            with open(img_list_txt, 'r') as f:
                for line in f.readlines():
                    self.img_list.append(line.strip('\n'))

        else:
            img_list_txt = os.path.join(self.imagesets_dir, mode + '.txt')
            with open(img_list_txt, 'r') as f:
                for line in f.readlines():
                    self.img_list.append(line.strip('\n'))

        self.trans = trans

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img_path = os.path.join(self.image_dir, self.img_list[item] + '.png')
        img_pil = Image.open(img_path).convert('L')
        img = img_pil.copy()
        scale_ratio = 32 / img.size[1]
        img = img.resize(size=(int(img.size[0] * scale_ratio), 32), resample=2)
        if img.size[0] > 164:
            img = img.resize(size=(164, 32), resample=2)
        else:
            pass
        img = np.array(img)
        # 哎，我就是不用opencv
        if img.shape[1] < 164:
            pad_num = 164 - img.shape[1]
            img = np.pad(img, ((0, 0), (pad_num // 2, pad_num - pad_num // 2)), constant_values=(0, 0))
        label_path = os.path.join(self.label_dir, self.img_list[item] + '.txt')
        label_origin = np.loadtxt(label_path)
        label = label_origin.copy()
        # 建议先合并 在填充，在切分
        label_length = label_origin.shape[0]
        if label_length < 40:
            pad_num = 40 - label_length
            label = np.pad(label, (0, pad_num), constant_values=(0, 0))
        image_length = 40
        image = torch.tensor(img, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.float)
        return image, image_length, label, label_length


if __name__ == '__main__':
    data_dir = "D:\\PythonProject\\CRNN\\data_alpha"
    dataset = MyDataSet(data_dir, mode='train')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, drop_last=False)
    for batch, data_couple in enumerate(dataloader):
        print("batch: ", batch)
        image, image_length, label, label_length = data_couple
        image_np = image.numpy()
        print("image's size: ", image.size())
        print("image_label: ", image_length)
        print("label's size: ", label.size())
        print("label's length: ", label_length)
