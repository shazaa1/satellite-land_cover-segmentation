
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

from deeplabv3plus import DeepLabV3Plus


CLASS_NAMES = [
    "Background",
    "Buildings",
    "Woodlands",
    "Water",
    "Roads"
]

# Visualization colors (RGB)
CLASS_COLORS = [
    (0, 0, 0),       # Background
    (255, 0, 0),     # Buildings
    (0, 255, 0),     # Woodlands
    (0, 0, 255),     # Water
    (255, 255, 0)    # Roads
]


class DeepLabPredictor:

    def __init__(self, weights_path, device=None):

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = DeepLabV3Plus(
            num_classes=5,
            pretrained=False
        )

        weights = torch.load(
            weights_path,
            map_location="cpu",
            weights_only=True
        )

        self.model.load_state_dict(weights, strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.Resize(
                (512, 512),
                interpolation=InterpolationMode.BILINEAR
            ),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @torch.no_grad()
    def predict(self, image_path):

        image = Image.open(image_path).convert("RGB")
        original_size = image.size

        input_tensor = self.transform(image).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)

        logits = self.model(input_tensor)
        prediction = torch.argmax(logits, dim=1)[0]

        mask = prediction.cpu().numpy().astype(np.uint8)

        # Resize predicted mask back to original image dimensions
        mask_image = Image.fromarray(mask)
        mask_image = mask_image.resize(
            original_size,
            resample=Image.Resampling.NEAREST
        )

        return np.array(mask_image)

    def colorize(self, mask):

        colored_mask = np.zeros(
            (*mask.shape, 3),
            dtype=np.uint8
        )

        for class_id, color in enumerate(CLASS_COLORS):
            colored_mask[mask == class_id] = color

        return Image.fromarray(colored_mask)
