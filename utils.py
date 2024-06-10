from typing import Dict, Any, Tuple
from dataclasses import dataclass

import pprint
import requests

import torch
import matplotlib.pyplot as plt
from PIL import Image

from model import Predictions, SimpleDETR

pp = pprint.PrettyPrinter(indent=4)

# COCO classes
CLASSES = [
    "N/A",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# colors for visualization
COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933],
]


@dataclass
class Boxes:
    """Data class for bounding boxes."""

    loc: torch.Tensor
    scores: torch.Tensor

    def __post_init__(self):
        self.label = self.scores.argmax(dim=1)

    def rescale(self, size: Tuple[int, int]):
        """Rescale to image dimensions."""
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(self.loc)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        self.loc = b

    def box_cxcywh_to_xyxy(self, x):
        """for output bounding box post-processing."""
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)


def visualize(pil_img, boxes_: Boxes):
    """Visualize bounding boxes over images."""
    prob = boxes_.scores
    boxes = boxes_.loc
    cls = boxes_.label
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, cl, (xmin, ymin, xmax, ymax), c in zip(
        prob, cls, boxes.tolist(), COLORS * 100
    ):
        ax.add_patch(
            plt.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=3
            )
        )
        # cl = p.argmax()
        text = f"{CLASSES[cl]}: {p[cl]:0.2f}"
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor="yellow", alpha=0.5))
    plt.axis("off")
    plt.show()


def map_weights(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Map the model's original weights to the new definition."""
    # pp.pprint(list(state_dict.keys()))
    new_state_dict = {}
    for k, v in state_dict.items():
        if "transformer.encoder" in k:
            new_state_dict[k] = v
        if "transformer.decoder" in k:
            new_state_dict[k] = v
        # if "transformer" in k:
        #     k = f"transformer.{k}"
        new_state_dict[k] = v
    return new_state_dict


def get_image_from_url(url: str) -> Image.Image:
    """Get Image from a URL."""

    return Image.open(requests.get(url, stream=True, timeout=10.0).raw)
