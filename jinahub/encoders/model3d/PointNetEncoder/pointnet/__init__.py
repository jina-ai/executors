import os

from models.pointnet_sem_seg import get_model
import torch


cur_dir = os.path.dirname(os.path.abspath(__file__))
NUM_CLASSES = 13

model = get_model(NUM_CLASSES).cuda()
