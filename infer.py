import logging
from typing import Callable

from PIL import Image
import torch
import torchvision.transforms as T
from rich.logging import RichHandler

from simple_detr.utils import Boxes, map_official_weights, visualize, get_image_from_url
from simple_detr.model import SimpleDETR, Predictions

torch.set_grad_enabled(False)


class Detector:
    """2D Object detector."""

    def __init__(self, threshold: float = 0.7, device: str = "cpu") -> None:
        self.device = device
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
            map_location=self.device,
            check_hash=True,
        )
        state_dict = map_official_weights(state_dict)
        self.detr.load_state_dict(state_dict)
        self.detr.to(self.device)
        self.detr.eval()
        self.threshold = threshold

    def preprocess(self, image_rgb: Image.Image) -> torch.Tensor:
        """Preprocess image and convert to tensor

        Args:
            image_rgb (Image.Image): RGB image object

        Returns:
            torch.Tensor: normalized image tensor.
        """
        # mean-std normalize the input image (batch-size: 1)
        return self.transform(image_rgb).unsqueeze(0)

    def detect(self, img: torch.Tensor) -> Boxes:
        """Detect object and visualize."""

        # demo model only support by default images with aspect ratio between 0.5 and 2
        # if you want to use images with an aspect ratio outside this range
        # rescale your image so that the maximum size is at most 1333 for best results
        assert (
            img.shape[-2] <= 1600 and img.shape[-1] <= 1600
        ), "demo model only supports images up to 1600 pixels on each side"

        img_ = img.to(self.device)

        # propagate through the model
        outputs: Predictions = self.detr(img_)

        # keep only predictions with 0.7+ confidence
        probas = outputs.logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > self.threshold

        boxes = Boxes(loc=outputs.boxes[0, keep].cpu(), scores=probas[keep].cpu())
        return boxes

    def visualize(self, image: Image.Image, boxes: Boxes) -> None:
        """Visualize boxes on image

        Args:
            image (Image.Image): RGB images.
            boxes (Boxes): box predictions.
        """
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

    detector = Detector(device="cuda")

    query_image = get_image_from_url(
        url="https://media.wired.com/photos/593256b42a990b06268a9e21/master/pass/traffic-jam-getty.jpg"
        # url="http://images.cocodataset.org/val2017/000000039769.jpg"
    )
    try:
        input_tensor = detector.preprocess(query_image)
        predictions = detector.detect(input_tensor)
        detector.visualize(query_image, predictions)
    except Exception as ex:
        logger.exception(ex)
