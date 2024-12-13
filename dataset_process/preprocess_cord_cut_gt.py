import json
import os

import numpy as np
from PIL import Image

from XYcut import recursive_xy_cut


def string_box(box):
    return (
         str(box[0])
        +" "
        +str(box[1])
        +" "
        +str(box[2])
        +" "
        +str(box[3])
    )
def quad_to_box(quad):
    # test 87 is wrongly annotated
    box = (
        max(0, quad["x1"]),
        max(0, quad["y1"]),
        quad["x3"],
        quad["y3"]
    )
    if box[3] < box[1]:
        bbox = list(box)
        tmp = bbox[3]
        bbox[3] = bbox[1]
        bbox[1] = tmp
        box = tuple(bbox)
    if box[2] < box[0]:
        bbox = list(box)
        tmp = bbox[2]
        bbox[2] = bbox[0]
        bbox[0] = tmp
        box = tuple(bbox)
    return box

def normalize_bbox(bbox, width, length):
    return [
        int(1000 * bbox[0] / width),
        int(1000 * bbox[1] / length),
        int(1000 * bbox[2] / width),
        int(1000 * bbox[3] / length),
    ]

def get_line_bbox(bboxs):
    x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
    y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]

    x0, y0, x1, y1 = min(x), min(y), max(x), max(y)

    assert x1 >= x0 and y1 >= y0
    bbox = [[x0, y0, x1, y1] for _ in range(len(bboxs))]
    return bbox

ori_path='../../CORD/test/gpt3_test_gt'
files=os.listdir(ori_path)
res_path='../../CORD/test/gpt3_test_cut_gt'


for file in files:
    fpath=os.path.join(ori_path,file)
    with open(fpath,'r',encoding='utf-8') as f:
        data=f.read()
        data=data.split('}')[:-1]
    res=''
    x_min = 1000
    x_max = 0
    y_min = 1000
    y_max = 0
    for j in range(len(data)):
        b1=data[j].split('Box:')[-1].split(',')[0]
        b1=b1.replace('[','').replace(']','').split(' ')
        b1=[int(b) for b in b1]
        if b1[0]<x_min:
            x_min=b1[0]               
        if b1[2]>x_max:
            x_max=b1[2]
        if b1[1]<y_min:
            y_min=b1[1]
        if b1[3]>y_max:
            y_max=b1[3]    
    for j in range(len(data)):
        b1=data[j].split('Box:')[-1].split(',')[0]
        b1=b1.replace('[','').replace(']','').split(' ')
        b1=[int(b) for b in b1]
        b1[0]=b1[0]-x_min+10
        b1[1]=b1[1]-y_min+10
        b1[2]=b1[2]-x_min+10
        b1[3]=b1[3]-y_min+10
        text=data[j].split('Box:')[0]+'Box:['+str(b1[0])+' '+str(b1[1])+' '+str(b1[2])+' '+str(b1[3])+'],'+data[j].split(',')[-1] +'}'
        # For ocr data, use the following line instead
        # text=data[j].split('Box:')[0]+'Box:['+str(b1[0])+' '+str(b1[1])+' '+str(b1[2])+' '+str(b1[3])+']}'
        res+=text
    with open(os.path.join(res_path,file),'w',encoding='utf-8') as f:
        f.write(res)