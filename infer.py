import logging
from typing import Callable

from PIL import Image
import torch
import torchvision.transforms as T
from rich.logging import RichHandler

from utils import Boxes, map_official_weights, visualize, get_image_from_url
from model import SimpleDETR, Predictions

torch.set_grad_enabled(False)


class Detector:
    """2D Object detector."""

    def __init__(self, threshold: float = 0.7) -> None:
        self.transform: Callable[[Image.Image], torch.Tensor] = T.Compose(
            [
                T.Resize(800),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.detr = SimpleDETR(num_classes=91)

        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth",
            map_location="cpu",
            check_hash=True,
        )

        state_dict = map_official_weights(state_dict)
        self.detr.load_state_dict(state_dict)
        self.detr.to("cpu")
        self.detr.eval()
        self.threshold = threshold

    def detect_and_visualize(self, image: Image.Image) -> None:
        """Detect object and visualize."""

        # mean-std normalize the input image (batch-size: 1)
        img = self.transform(image).unsqueeze(0)

        # demo model only support by default images with aspect ratio between 0.5 and 2
        # if you want to use images with an aspect ratio outside this range
        # rescale your image so that the maximum size is at most 1333 for best results
        assert (
            img.shape[-2] <= 1600 and img.shape[-1] <= 1600
        ), "demo model only supports images up to 1600 pixels on each side"

        # propagate through the model
        outputs: Predictions = self.detr(img)

        # keep only predictions with 0.7+ confidence
        probas = outputs.logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > self.threshold

        boxes = Boxes(loc=outputs.boxes[0, keep], scores=probas[keep])
        boxes.rescale(image.size)

        visualize(image, boxes)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    logger = logging.getLogger("main")
    logger.info("Object detection")

    detector = Detector()

    # url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    # url = "https://media.wired.com/photos/593256b42a990b06268a9e21/master/pass/traffic-jam-getty.jpg"
    # url = "https://mcity.umich.edu/wp-content/uploads/2024/02/Birmingham-Intersection.jpg"

    query_image = get_image_from_url(
        "https://media.wired.com/photos/593256b42a990b06268a9e21/master/pass/traffic-jam-getty.jpg"
    )
    try:
        detector.detect_and_visualize(query_image)
    except Exception as ex:
        logger.exception(ex)
