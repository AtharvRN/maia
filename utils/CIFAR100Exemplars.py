import base64
import io
import os
from typing import Dict, Iterable, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm.auto import tqdm


class CIFAR100Exemplars:
    """
    Lightweight MAIA-compatible exemplar loader for classroom demos.

    This class mimics the small subset of DatasetExemplars used by
    Tools.dataset_exemplars:

        self.exemplars[layer][unit] -> list[base64_png]
        self.activations[layer][unit] -> np.ndarray[n_exemplars]
        self.thresholds[layer][unit] -> float

    Unlike the original DatasetExemplars class, this computes exemplars live from
    CIFAR-100 instead of loading NetDissect/ImageNet files.
    """

    def __init__(
        self,
        path2save: str,
        model_name: str = "resnet50",
        layers: str | List[str] = "layer4",
        units: Iterable[int] | None = None,
        n_exemplars: int = 15,
        probe_size: int = 1000,
        batch_size: int = 128,
        device: str | int = 0,
        data_root: str = "./data",
    ):
        self.path2save = path2save
        self.model_name = model_name
        self.layers = layers if isinstance(layers, list) else [layers]
        self.units = list(units) if units is not None else None
        self.n_exemplars = n_exemplars
        self.probe_size = probe_size
        self.batch_size = batch_size
        self.data_root = data_root
        self.device = torch.device(
            f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        )

        self.exemplars: Dict[str, List[List[str]]] = {}
        self.activations: Dict[str, np.ndarray] = {}
        self.thresholds: Dict[str, np.ndarray] = {}

        self.model, self.preprocess = self._load_model_and_preprocess(model_name)
        self.raw_dataset = datasets.CIFAR100(
            root=self.data_root, train=False, download=True, transform=None
        )
        self.tensor_dataset = datasets.CIFAR100(
            root=self.data_root, train=False, download=True, transform=self.preprocess
        )
        subset_size = min(self.probe_size, len(self.tensor_dataset))
        self.indices = list(range(subset_size))
        self.tensor_subset = Subset(self.tensor_dataset, self.indices)

        for layer in self.layers:
            exemplars, activations, thresholds = self.compute_layer_exemplars(layer)
            self.exemplars[layer] = exemplars
            self.activations[layer] = activations
            self.thresholds[layer] = thresholds

    def _load_model_and_preprocess(self, model_name):
        if model_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            model = models.resnet50(weights=weights).to(self.device).eval()
            return model, weights.transforms()
        if model_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            model = models.resnet18(weights=weights).to(self.device).eval()
            return model, weights.transforms()
        raise ValueError(f"Unsupported demo model: {model_name}")

    def _get_module(self, dotted_name):
        module = self.model
        for part in dotted_name.split("."):
            module = getattr(module, part)
        return module

    @staticmethod
    def _pool_activations(output):
        if output.ndim == 4:
            return output.flatten(2).max(dim=2).values
        if output.ndim == 2:
            return output
        raise ValueError(f"Unsupported activation shape: {tuple(output.shape)}")

    @staticmethod
    def _pil_to_base64_png(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _save_and_encode_image(self, layer, unit, rank, image):
        save_dir = os.path.join(
            self.path2save,
            "cifar100_exemplars",
            self.model_name,
            layer,
            str(unit),
        )
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{rank}.png")
        image.save(save_path)
        return self._pil_to_base64_png(image)

    @torch.no_grad()
    def compute_layer_exemplars(self, layer):
        captured = []

        def hook(_module, _inputs, output):
            captured.append(self._pool_activations(output.detach()).cpu())

        handle = self._get_module(layer).register_forward_hook(hook)
        all_acts = []
        loader = DataLoader(
            self.tensor_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=(self.device.type == "cuda"),
        )

        try:
            for images, _labels in tqdm(loader, desc=f"CIFAR100 {layer} activations"):
                _ = self.model(images.to(self.device, non_blocking=True))
                all_acts.append(captured.pop())
        finally:
            handle.remove()

        acts = torch.cat(all_acts, dim=0)
        n_units = acts.shape[1]
        unit_ids = self.units if self.units is not None else list(range(n_units))

        exemplar_images: List[List[str]] = [[] for _ in range(n_units)]
        activation_table = np.zeros((n_units, self.n_exemplars), dtype=np.float32)
        thresholds = np.zeros(n_units, dtype=np.float32)

        for unit in unit_ids:
            if unit < 0 or unit >= n_units:
                raise ValueError(f"Unit {unit} is out of range for {layer} ({n_units} units).")
            values, local_ids = acts[:, unit].topk(k=min(self.n_exemplars, len(acts)))
            thresholds[unit] = float(values.float().mean())
            activation_table[unit, : len(values)] = values.numpy()

            encoded_images = []
            for rank, local_idx in enumerate(local_ids.tolist()):
                raw_idx = self.indices[local_idx]
                image, _label = self.raw_dataset[raw_idx]
                image = image.resize((224, 224), Image.Resampling.BICUBIC)
                encoded_images.append(self._save_and_encode_image(layer, unit, rank, image))
            exemplar_images[unit] = encoded_images

        return exemplar_images, activation_table, thresholds
