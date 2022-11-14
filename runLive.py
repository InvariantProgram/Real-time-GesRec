import os
import sys
import json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import cv2
from utils import *

from models import mobilenet

if __name__ == '__main__':
    gestures = open("annotation_Jester/categories.txt").readlines()
    gestures = [gesture.split()[-1] for gesture in gestures]
    print(gestures)

    pretrained = torch.load("results/jester_mobilenet_0.5x_RGB_16_best.pth")
    #dataset_architect_[width_mult]x_modality_sampleDuration 
    
    '''
    for key in pretrained:
        print(key, type(pretrained[key]))

    print(pretrained["arch"])
    '''
    model = mobilenet.get_model(num_classes = 27, width_mult = 0.5)
    model = nn.DataParallel(model, device_ids=None)
    model.module.classifier = nn.Sequential(
                                nn.Dropout(0.5),
                                nn.Linear(model.module.classifier[1].in_features, 27))
    model = model.cuda()
    model.eval()


    print("Starting Video Process")
    cv2.namedWindow("preview")

    q = list()
    max_size = 16 # Number of frames stored
    vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False
    
    while rval:
        if len(q) == max_size:
            test = q.pop() #numpy array (480, 640, 3)
        q.insert(0, torch.Tensor(frame))


        gest_str = ""
        if len(q) == max_size:
            with torch.no_grad():
                input = torch.unsqueeze(torch.stack(q, 0).permute(3, 0, 1, 2), 0)
                output = F.softmax(model(input), dim=1) #torch.size([1, 27])
                print(output)
                max_ind = torch.argmax(output).item()
                gest_str = "Gesture: %s" % (gestures[max_ind])

        rval, frame = vc.read()
        cv2.putText(frame, gest_str, (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.imshow("preview", frame)

        key = cv2.waitKey(20)
        if key == 27:
            break

    cv2.destroyWindow("preview")
    vc.release()
    