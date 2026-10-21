
"""Notebook-faithful CoralScope inference with sequential model loading.

EfficientNetV2-S + ConvNeXt-Tiny, fold 0, 3-view TTA.
Only one classification model is kept in memory at a time.
"""
from pathlib import Path
from io import BytesIO
import base64
import gc

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image

from model import build_model


# Reduce CPU thread overhead on resource-constrained hosting.
torch.set_num_threads(1)

CLASSES = ["bleached", "dead", "healthy"]

MODEL_NAMES = [
    "tf_efficientnetv2_s.in21k_ft_in1k",
    "convnext_tiny.fb_in22k_ft_in1k",
]

IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Notebook validation transform and three-view TTA.
VAL_TRANSFORM = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

TTA_TRANSFORMS = [
    VAL_TRANSFORM,
    A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=1.0),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ]),
    A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.RandomRotate90(p=1.0),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ]),
]


def underwater_color_correct(img_bgr: np.ndarray) -> np.ndarray:
    """Apply notebook CLAHE to the L channel in LAB color space."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8),
    )

    corrected_lab = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)


def _b64_jpeg(rgb: np.ndarray, quality: int = 90) -> str:
    buff = BytesIO()
    Image.fromarray(rgb).save(buff, format="JPEG", quality=quality)
    return base64.b64encode(buff.getvalue()).decode("ascii")


class CoralPredictor:
    """Sequential two-backbone softmax ensemble with three-view TTA.

    Models are loaded for an operation and released afterward.
    Grad-CAM is computed with EfficientNetV2-S when requested.
    """

    def __init__(self, checkpoint_dir: str | Path, fold: int = 0):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.fold = int(fold)
        self.models = {}

        # Validate both files without loading either model into RAM.
        missing = [
            str(self._checkpoint_path(name))
            for name in MODEL_NAMES
            if not self._checkpoint_path(name).is_file()
        ]

        if missing:
            raise FileNotFoundError(
                "The ensemble requires both fold checkpoints. Missing: "
                + ", ".join(missing)
            )

        self.checkpoint = ";".join(
            self._checkpoint_path(name).name for name in MODEL_NAMES
        )

    def _checkpoint_path(self, name: str) -> Path:
        filename = f"{name.replace('/', '_')}_fold{self.fold}.pt"
        return self.checkpoint_dir / filename

    def _load_model(self, name: str):
        """Load a single checkpoint, preferring memory-efficient assignment."""
        path = self._checkpoint_path(name)

        model = build_model(
            name,
            num_classes=len(CLASSES),
            dropout=0.3,
        )

        try:
            state = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            state = torch.load(path, map_location="cpu")

        try:
            # assign=True can avoid copying checkpoint tensors into a second
            # set of initialized parameter tensors on supported PyTorch.
            try:
                model.load_state_dict(state, assign=True)
            except TypeError:
                model.load_state_dict(state)
        finally:
            del state

        model.to(DEVICE)
        model.eval()
        return model

    def _prepare(self, image_bgr: np.ndarray):
        corrected_bgr = underwater_color_correct(image_bgr)
        corrected_rgb = cv2.cvtColor(
            corrected_bgr,
            cv2.COLOR_BGR2RGB,
        )
        return corrected_bgr, corrected_rgb

    def _predict_one_model(
        self,
        model_name: str,
        corrected_rgb: np.ndarray,
    ) -> np.ndarray:
        """Average the three TTA softmax vectors for one model."""
        model = None

        try:
            model = self._load_model(model_name)
            probs_sum = torch.zeros(len(CLASSES), device=DEVICE)

            with torch.inference_mode():
                for transform in TTA_TRANSFORMS:
                    x = transform(
                        image=corrected_rgb
                    )["image"].unsqueeze(0).to(DEVICE)

                    with torch.autocast(
                        device_type=DEVICE.type,
                        enabled=(DEVICE.type == "cuda"),
                    ):
                        logits = model(x)

                    probs_sum += F.softmax(logits.float(), dim=1)[0]

            return (probs_sum / len(TTA_TRANSFORMS)).cpu().numpy()

        finally:
            if model is not None:
                del model

            gc.collect()

            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    def _predict_probs(self, corrected_rgb: np.ndarray) -> np.ndarray:
        """Average both models' three-view probability vectors."""
        ensemble_sum = np.zeros(len(CLASSES), dtype=np.float64)

        for model_name in MODEL_NAMES:
            model_probs = self._predict_one_model(
                model_name,
                corrected_rgb,
            )
            ensemble_sum += model_probs

            del model_probs
            gc.collect()

        return (ensemble_sum / len(MODEL_NAMES)).astype(np.float32)

    @staticmethod
    def _uncertainty(probs: np.ndarray) -> dict:
        sorted_p = np.sort(probs)[::-1]
        margin = float(sorted_p[0] - sorted_p[1])
        entropy = float(
            -(probs * np.log(probs + 1e-12)).sum()
            / np.log(len(probs))
        )
        confidence = float(sorted_p[0])

        needs_review = (
            confidence < 0.70
            or margin < 0.15
            or entropy > 0.75
        )

        reasons = []
        if confidence < 0.70:
            reasons.append("top-class score below 0.70")
        if margin < 0.15:
            reasons.append("top-two class margin below 0.15")
        if entropy > 0.75:
            reasons.append("normalized entropy above 0.75")

        return {
            "confidence": confidence,
            "top_two_margin": margin,
            "normalized_entropy": entropy,
            "needs_review": needs_review,
            "review_reasons": reasons,
        }

    def predict_bgr(
        self,
        image_bgr: np.ndarray,
        source: str = "unspecified",
        explain: bool = True,
    ) -> dict:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("The uploaded file is not a valid image.")

        _, corrected_rgb = self._prepare(image_bgr)
        probs = self._predict_probs(corrected_rgb)

        idx = int(np.argmax(probs))
        uncertainty = self._uncertainty(probs)

        result = {
            "predicted_class": CLASSES[idx],
            "confidence": uncertainty["confidence"],
            "probabilities": {
                label: float(probs[i])
                for i, label in enumerate(CLASSES)
            },
            "model": (
                "EfficientNetV2-S + ConvNeXt-Tiny ensemble + 3-view TTA"
            ),
            "backbones": MODEL_NAMES,
            "checkpoint": self.checkpoint,
            "fold": self.fold,
            "device": str(DEVICE),
            "source": source,
            **uncertainty,
            "preprocessing": (
                "LAB/CLAHE (clipLimit=2.5, grid=8x8), RGB, "
                "resize 224x224, ImageNet normalization"
            ),
            "tta_views": 3,
        }

        if explain:
            result["corrected_image"] = (
                f"data:image/jpeg;base64,{_b64_jpeg(corrected_rgb)}"
            )
            result.update(self._gradcam(corrected_rgb, idx))

        return result

    def _gradcam(
        self,
        corrected_rgb: np.ndarray,
        class_idx: int,
    ) -> dict:
        """Load only EfficientNetV2-S to compute the Grad-CAM heatmap."""
        model = None
        handle = None
        activations = {}
        gradients = {}

        try:
            model = self._load_model(MODEL_NAMES[0])

            target_layer = (
                model.backbone.blocks[-1]
                if hasattr(model.backbone, "blocks")
                else model.backbone.stages[-1]
            )

            def forward_hook(_module, _inputs, output):
                activations["value"] = output
                if output.requires_grad:
                    output.register_hook(
                        lambda grad: gradients.__setitem__("value", grad)
                    )

            handle = target_layer.register_forward_hook(forward_hook)

            x = VAL_TRANSFORM(
                image=corrected_rgb
            )["image"].unsqueeze(0).to(DEVICE)

            model.zero_grad(set_to_none=True)

            with torch.enable_grad():
                with torch.autocast(
                    device_type=DEVICE.type,
                    enabled=(DEVICE.type == "cuda"),
                ):
                    logits = model(x)

                logits[0, class_idx].backward()

            feat = activations["value"].detach()
            grad = gradients["value"].detach()

            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu(
                (weights * feat).sum(dim=1, keepdim=True)
            )

            cam = F.interpolate(
                cam,
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="bilinear",
                align_corners=False,
            )[0, 0]

            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)

            gray = (cam.cpu().numpy() * 255).astype(np.uint8)
            heat_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

            return {
                "gradcam_heatmap": (
                    f"data:image/jpeg;base64,{_b64_jpeg(heat_rgb)}"
                ),
                "explanation": (
                    "Grad-CAM is computed on the primary "
                    "EfficientNetV2-S model for the class selected by "
                    "the ensemble. It is a qualitative diagnostic, "
                    "not segmentation or causal proof."
                ),
            }

        finally:
            if handle is not None:
                handle.remove()

            if model is not None:
                model.zero_grad(set_to_none=True)
                del model

            gc.collect()

            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
