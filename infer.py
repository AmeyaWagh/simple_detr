from PIL import Image
import requests
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T

from utils import *
from model import *

torch.set_grad_enabled(False)


detr = CustomDETR(num_classes=91)
state_dict = torch.hub.load_state_dict_from_url(
    url="https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth",
    map_location="cpu",
    check_hash=True,
)

state_dict = map_weights(state_dict)
detr.load_state_dict(state_dict)
detr.eval()


# standard PyTorch mean-std input image normalization
transform = T.Compose(
    [
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
url = "https://media.wired.com/photos/593256b42a990b06268a9e21/master/pass/traffic-jam-getty.jpg"
# url = "https://mcity.umich.edu/wp-content/uploads/2024/02/Birmingham-Intersection.jpg"
im = Image.open(requests.get(url, stream=True).raw)

scores, boxes = detect(im, detr, transform)


plot_results(im, scores, boxes)
