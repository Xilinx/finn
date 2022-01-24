import os
from re import sub
import shutil
import xmltodict
import numpy as np
from tqdm import tqdm

data_dir = "/home/atchelet/Downloads/data_training/"
out_dir = "/home/atchelet/Dataset"
os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(out_dir, "labels"), exist_ok=True)

w_px = h_px = 40

for path, subdirs, files in os.walk(data_dir):
    dir = os.path.basename(path)
    i = 0
    for file in tqdm(sorted(files), desc=dir):
        if file.endswith('.jpg'):
            shutil.copyfile(os.path.join(path, file), os.path.join(
                out_dir, "images", f"{dir}_{i:07d}.jpg"))
        if file.endswith('.xml'):
            with open(os.path.join(path, file)) as fd:
                doc = xmltodict.parse(fd.read())
                width = int(doc['annotation']['size']['width'])
                height = int(doc['annotation']['size']['height'])
                xmin = int(doc['annotation']['object']['bndbox']['xmin'])
                ymin = int(doc['annotation']['object']['bndbox']['ymin'])
                xmax = int(doc['annotation']['object']['bndbox']['xmax'])
                ymax = int(doc['annotation']['object']['bndbox']['ymax'])
                b_x = (((xmax + xmin) / 2) / width)
                b_y = (((ymax + ymin) / 2) / height)
                b_w = (xmax - xmin) / width
                b_h = (ymax - ymin) / height
                f = open(os.path.join(out_dir, "labels",
                         f"{dir}_{i:07d}.txt"), "a")
                f.write(f"{dir}_{i:07d}\n{b_x}\t{b_y}\t{b_w}\t{b_h}\n")
                f.close()
                i += 1
