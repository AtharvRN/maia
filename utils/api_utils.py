import base64
import io
import os

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def base64_to_url(value: str) -> str:
    if value.startswith("data:image"):
        return value
    return f"data:image/png;base64,{value}"


def is_base64(value) -> bool:
    if not isinstance(value, str):
        return False
    if value.startswith("data:image"):
        return True
    try:
        base64.b64decode(value, validate=True)
        return True
    except Exception:
        return False


def image2str(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def str2image(value: str) -> Image.Image:
    if value.startswith("data:image"):
        value = value.split(",", 1)[1]
    raw = base64.b64decode(value)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def format_api_content(type: str, input: str):
    if type == "text":
        return {"type": "text", "text": str(input)}
    if type == "image_url":
        return {"type": "image_url", "image_url": {"url": base64_to_url(input)}}
    raise ValueError(f"Unsupported type: {type}")


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().cpu().float()
    if image.ndim == 4:
        image = image[0]
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    image = image * std + mean
    image = image.clamp(0, 1)
    return transforms.ToPILImage()(image)


def generate_masked_image(image, mask, path2save="./temp.png", threshold=0):
    pil = _tensor_to_pil(image)
    mask = mask.detach().cpu().float()
    if mask.ndim == 3:
        mask = mask[0]
    mask = mask.unsqueeze(0).unsqueeze(0)
    mask = F.interpolate(mask, size=pil.size[::-1], mode="bilinear", align_corners=False)[0, 0]
    if threshold is None:
        threshold = float(mask.mean())
    keep = (mask >= threshold).float().clamp(0, 1)
    keep = keep.unsqueeze(0).repeat(3, 1, 1)

    img_t = transforms.ToTensor()(pil)
    darkened = img_t * keep + 0.25 * img_t * (1 - keep)
    out = transforms.ToPILImage()(darkened.clamp(0, 1))

    os.makedirs(os.path.dirname(path2save) or ".", exist_ok=True)
    out.save(path2save)
    return image2str(out)
