"""Dataset loading.

Two families, matching the official AReS data pipeline:

* LMDB-packaged datasets with CoOp splits (from ILM-VP, re-hosted by the AReS
  authors): flowers102, dtd, eurosat, oxfordpets.
* plain torchvision datasets with official test splits: svhn, gtsrb.

Stage 1 (priming) feeds raw images through the CLIP preprocess. Stage 2
(reprogramming) resizes images to a small source size; the learnable padding
prompt fills the rest of the 224x224 canvas (ILM-VP style).
"""

import io
import json
import os
import pickle
import random

import lmdb
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets as tvd
from torchvision import transforms

from templates import GTSRB_CLASSES

LMDB_DATASETS = ["flowers102", "dtd", "eurosat", "oxfordpets"]
ALL_DATASETS = LMDB_DATASETS + ["svhn", "gtsrb"]

# image size the target images are resized to before padding (ILM-VP)
SOURCE_SIZE = {ds: 128 for ds in LMDB_DATASETS}
SOURCE_SIZE.update({"svhn": 32, "gtsrb": 32})

# training batch sizes from the official code
TRAIN_BATCH = {ds: 256 for ds in ALL_DATASETS}
TRAIN_BATCH.update({"dtd": 64, "oxfordpets": 64})


def refine_classnames(class_names):
    return [c.lower().replace("_", " ").replace("-", " ") for c in class_names]


class CoopLMDB(Dataset):
    """Reader for the CoOp-split LMDB datasets.

    The lmdb environment is opened lazily so the dataset object can be
    pickled into DataLoader workers on Windows (spawn start method).
    """

    def __init__(self, root, split, transform=None):
        self.db_path = os.path.join(root, f"{split}.lmdb")
        self.transform = transform
        self.env = None
        env = lmdb.open(self.db_path, subdir=os.path.isdir(self.db_path),
                        readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b"__len__"))
            self.keys = pickle.loads(txn.get(b"__keys__"))
        env.close()
        with open(os.path.join(root, "split.json")) as f:
            split_file = json.load(f)
        idx_to_class = dict(sorted({s[-2]: s[-1] for s in split_file["test"]}.items()))
        self.classes = list(idx_to_class.values())

    def _open(self):
        self.env = lmdb.open(self.db_path, subdir=os.path.isdir(self.db_path),
                             readonly=True, lock=False, readahead=False, meminit=False)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.env is None:
            self._open()
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = pickle.loads(byteflow)
        img = Image.open(io.BytesIO(unpacked[0]))
        target = unpacked[1]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def scan_labels(self):
        # label-only pass, avoids decoding images
        if self.env is None:
            self._open()
        labels = []
        with self.env.begin(write=False) as txn:
            for k in self.keys:
                labels.append(pickle.loads(txn.get(k))[1])
        return labels


def get_dataset(name, split, data_dir, transform):
    """Returns (dataset, class_names). split is 'train' or 'test'."""
    root = os.path.join(data_dir, name)
    if name in LMDB_DATASETS:
        ds = CoopLMDB(root, split, transform)
        classes = refine_classnames(ds.classes)
    elif name == "svhn":
        ds = tvd.SVHN(root, split=split, download=True, transform=transform)
        classes = [str(i) for i in range(10)]
    elif name == "gtsrb":
        ds = tvd.GTSRB(root, split=split, download=True, transform=transform)
        classes = refine_classnames(list(GTSRB_CLASSES))
    else:
        raise ValueError(name)
    return ds, classes


def dataset_labels(ds):
    if isinstance(ds, CoopLMDB):
        return ds.scan_labels()
    if isinstance(ds, tvd.SVHN):
        return list(ds.labels)
    if isinstance(ds, tvd.GTSRB):
        return [s[1] for s in ds._samples]
    return [ds[i][1] for i in range(len(ds))]


def few_shot_indices(ds, n_per_class, seed):
    """Same sampling procedure as the official code: group indices by label in
    order of first appearance, then random.sample(indices, n) per class."""
    class_to_indices = {}
    for i, label in enumerate(dataset_labels(ds)):
        class_to_indices.setdefault(label, []).append(i)
    rng = random.Random(seed)
    picked = []
    for indices in class_to_indices.values():
        if len(indices) >= n_per_class:
            picked.extend(rng.sample(indices, n_per_class))
        else:
            picked.extend(indices)
    return picked


def clip_transform(preprocess, name):
    # CLIP's own preprocess; LMDB images may be grayscale so force RGB first
    if name in LMDB_DATASETS:
        return transforms.Compose([transforms.Lambda(lambda x: x.convert("RGB")), preprocess])
    return preprocess


def vr_transform(name):
    """Transform for the reprogramming stage (no augmentation, like ILM-VP)."""
    size = SOURCE_SIZE[name]
    ops = []
    if name in LMDB_DATASETS:
        ops.append(transforms.Lambda(lambda x: x.convert("RGB")))
    if name != "svhn":  # svhn is already 32x32
        ops.append(transforms.Resize((size, size)))
    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)


def materialize(ds, num_workers=4, batch_size=256):
    """Decode a dataset once into an in-memory TensorDataset.

    All transforms here are deterministic, so this is exact and saves us from
    re-decoding jpegs for 200 eval epochs.
    """
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    return TensorDataset(torch.cat(xs), torch.cat(ys))
