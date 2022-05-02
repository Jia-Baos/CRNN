# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/28 15:27
# @Author  : Jis-Baos
# @File    : test.py

import os
import numpy as np

# 映射表
mapping_tables = ['-', 'A', 'a', 'B', 'b', 'C', 'c', 'D', 'd', 'E', 'e', 'F', 'f', 'G', 'g',
                  'H', 'h', 'I', 'i', 'J', 'j', 'K', 'k', 'L', 'l', 'M', 'm', 'N', 'n',
                  'O', 'o', 'P', 'p', 'Q', 'q', 'R', 'r', 'S', 's', 'T', 't',
                  'U', 'u', 'V', 'v', 'W', 'w', 'X', 'x', 'Y', 'y', 'Z', 'z']

data_path = "D:\\PythonProject\\CRNN\\data_alpha\\Labels\\word_14.txt"
save_path = "D:\\PythonProject\\CRNN\\test.txt"

with open(data_path, mode='r', encoding='utf_8') as f1:
    content = f1.readlines()
    print(content[0])
    for item in content[0]:
        print(item)
        if item in mapping_tables:
            alpha_index = mapping_tables.index(item)
        with open(save_path, mode='a+', encoding='utf-8') as f2:
            f2.write(str(alpha_index))
            f2.write(' ')