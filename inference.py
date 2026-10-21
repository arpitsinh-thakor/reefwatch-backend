
"""Notebook-faithful CoralScope inference:
EfficientNetV2-S + ConvNeXt-Tiny, fold 0, 3-view TTA.
"""
from pathlib import Path
from io import BytesIO
import base64

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image

from model import build_model


# Limit CPU thread usage on resource-constrained hosting.
torch.set_num_threads(1)

# Exact class order from notebook cell 15.
CLASSES = ["bleached", "dead", "healthy"]

MODEL_NAMES = [
    "tf_efficientnetv2_s.in21k_ft_in1k",
    "convnext_tiny.fb_in22k_ft_in1k",
]

IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Match notebook val_tf and tta_tf_list:
# identity, horizontal flip, and RandomRotate90.
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
    """Apply notebook CLAHE on the LAB L channel."""
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
    Image.fromarray(rgb).save(
        buff,
        format="JPEG",
        quality=quality,
    )
    return base64.b64encode(buff.getvalue()).decode("ascii")


class CoralPredictor:
    """Two-backbone softmax ensemble averaged over three TTA views.

    The deployed default is fold 0, matching the notebook demo.
    Grad-CAM is computed using the primary EfficientNetV2-S model.
    """

    def __init__(self, checkpoint_dir: str | Path, fold: int = 0):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.fold = int(fold)
        self.models = {}

        # Check that both checkpoints exist before loading either model.
        missing = []

        for name in MODEL_NAMES:
            filename = f"{name.replace('/', '_')}_fold{self.fold}.pt"
            path = self.checkpoint_dir / filename

            if not path.is_file():
                missing.append(str(path))

        if missing:
            raise FileNotFoundError(
                "The notebook-faithful ensemble requires both fold "
                "checkpoints. Missing: " + ", ".join(missing)
            )

        # Load the models one at a time.
        for name in MODEL_NAMES:
            path = self.checkpoint_dir / (
                f"{name.replace('/', '_')}_fold{self.fold}.pt"
            )

            model = build_model(
                name,
                num_classes=len(CLASSES),
                dropout=0.3,
            )

            # Load checkpoint tensors on CPU to avoid placing the
            # temporary state dictionary directly on the GPU.
            try:
                state = torch.load(
                    path,
                    map_location="cpu",
                    weights_only=True,
                )
            except TypeError:
                # Compatibility fallback for older PyTorch versions.
                state = torch.load(
                    path,
                    map_location="cpu",
                )

            model.load_state_dict(state)

            # Release the temporary checkpoint state dictionary.
            del state

            model.to(DEVICE)
            model.eval()

            self.models[name] = model

        self.checkpoint = ";".join(
            f"{name.replace('/', '_')}_fold{self.fold}.pt"
            for name in MODEL_NAMES
        )

    def _prepare(self, image_bgr: np.ndarray):
        corrected_bgr = underwater_color_correct(image_bgr)
        corrected_rgb = cv2.cvtColor(
            corrected_bgr,
            cv2.COLOR_BGR2RGB,
        )
        return corrected_bgr, corrected_rgb

    def _predict_probs(self, corrected_rgb: np.ndarray) -> np.ndarray:
        probs_sum = torch.zeros(
            len(CLASSES),
            device=DEVICE,
        )
        n = 0

        # Inference mode reduces autograd bookkeeping for classification.
        with torch.inference_mode():
            for model_name in MODEL_NAMES:
                model = self.models[model_name]

                for transform in TTA_TRANSFORMS:
                    x = transform(
                        image=corrected_rgb
                    )["image"].unsqueeze(0).to(DEVICE)

                    with torch.autocast(
                        device_type=DEVICE.type,
                        enabled=(DEVICE.type == "cuda"),
                    ):
                        logits = model(x)

                    probs_sum += F.softmax(
                        logits.float(),
                        dim=1,
                    )[0]

                    n += 1

        return (probs_sum / n).cpu().numpy()

    @staticmethod
    def _uncertainty(probs: np.ndarray) -> dict:
        sorted_p = np.sort(probs)[::-1]

        margin = float(sorted_p[0] - sorted_p[1])
        entropy = float(
            -(probs * np.log(probs + 1e-12)).sum()
            / np.log(len(probs))
        )
        confidence = float(sorted_p[0])

        # Operational review flag; scores are not calibrated ecological
        # probabilities.
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
            raise ValueError(
                "The uploaded file is not a valid image."
            )

        corrected_bgr, corrected_rgb = self._prepare(image_bgr)
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

        # Optional visual assets and Grad-CAM explanation.
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
        model = self.models[MODEL_NAMES[0]]

        target_layer = (
            model.backbone.blocks[-1]
            if hasattr(model.backbone, "blocks")
            else model.backbone.stages[-1]
        )

        activations = {}
        gradients = {}

        def forward_hook(_module, _inputs, output):
            activations["value"] = output

            if output.requires_grad:
                output.register_hook(
                    lambda grad: gradients.__setitem__("value", grad)
                )

        handle = target_layer.register_forward_hook(forward_hook)

        try:
            x = VAL_TRANSFORM(
                image=corrected_rgb
            )["image"].unsqueeze(0).to(DEVICE)

            model.zero_grad(set_to_none=True)

            # Grad-CAM requires gradients, so do not use inference_mode here.
            with torch.enable_grad():
                with torch.autocast(
                    device_type=DEVICE.type,
                    enabled=(DEVICE.type == "cuda"),
                ):
                    logits = model(x)

                logits[0, class_idx].backward()

            feat = activations["value"].detach()
            grad = gradients["value"].detach()

            weights = grad.mean(
                dim=(2, 3),
                keepdim=True,
            )

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

            gray = (
                cam.cpu().numpy() * 255
            ).astype(np.uint8)

            heat_bgr = cv2.applyColorMap(
                gray,
                cv2.COLORMAP_JET,
            )

            heat_rgb = cv2.cvtColor(
                heat_bgr,
                cv2.COLOR_BGR2RGB,
            )

            return {
                "gradcam_heatmap": (
                    f"data:image/jpeg;base64,{_b64_jpeg(heat_rgb)}"
                ),
                "explanation": (
                    "Grad-CAM is computed on the primary "
                    "EfficientNetV2-S fold-0 model for the class "
                    "selected by the ensemble. It is a qualitative "
                    "diagnostic, not segmentation or causal proof."
                ),
            }

        finally:
            handle.remove()
            model.zero_grad(set_to_none=True)
