"""Optional TorchScript adapter for a separately trained coral-damage segmentation model.

Checkpoint contract: TorchScript module accepting normalized RGB tensor [1,3,224,224] and
returning raw logits [1,1,H,W] (single damage foreground channel) or [1,2,H,W]
(background at channel 0, damage at channel 1). This adapter does not train a model.
"""
from pathlib import Path
import base64
from io import BytesIO

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from inference import underwater_color_correct, DEVICE, IMAGE_SIZE, MEAN, STD


class SegmentationPredictor:
    def __init__(self, checkpoint: str | Path):
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        self.model = torch.jit.load(str(self.checkpoint), map_location=DEVICE)
        self.model.to(DEVICE).eval()

    @staticmethod
    def _data_uri(image: np.ndarray, mode: str = "RGB") -> str:
        buff = BytesIO()
        Image.fromarray(image, mode=mode).save(buff, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buff.getvalue()).decode("ascii")

    @torch.no_grad()
    def predict_bgr(self, image_bgr: np.ndarray) -> dict:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("The uploaded file is not a valid image.")
        h, w = image_bgr.shape[:2]
        corrected = underwater_color_correct(image_bgr)
        rgb = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        normalized = (resized - np.asarray(MEAN, dtype=np.float32)) / np.asarray(STD, dtype=np.float32)
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
        logits = self.model(tensor)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if not torch.is_tensor(logits) or logits.ndim != 4 or logits.shape[0] != 1:
            raise ValueError("Segmentation checkpoint must return logits shaped [1,C,H,W].")
        if logits.shape[1] == 1:
            damage_prob = torch.sigmoid(logits)[0, 0]
        elif logits.shape[1] == 2:
            damage_prob = torch.softmax(logits, dim=1)[0, 1]
        else:
            raise ValueError("Segmentation checkpoint must return one foreground channel or [background, damage] channels.")
        damage_prob = F.interpolate(damage_prob[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        mask = (damage_prob >= 0.5).astype(np.uint8)
        original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        overlay = original_rgb.copy()
        overlay[mask == 1] = (235, 65, 50)
        overlay = cv2.addWeighted(original_rgb, 0.68, overlay, 0.32, 0)
        return {
            "available": True,
            "checkpoint": self.checkpoint.name,
            "damage_pixel_fraction": float(mask.mean()),
            "mean_damage_score": float(damage_prob.mean()),
            "mask_image": self._data_uri(mask * 255, mode="L"),
            "overlay_image": self._data_uri(overlay),
            "threshold": 0.5,
            "note": "Segmentation mask is only meaningful if this checkpoint was trained with matching CLAHE/224/ImageNet preprocessing and reviewed pixel-mask annotations. It is not available from the classification notebook alone.",
        }
