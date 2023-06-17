from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
import torch
import pytest

from matgl.utils.io import RemoteFile, get_available_pretrained_models, load_model, IOMixIn

this_dir = Path(os.path.abspath(os.path.dirname(__file__)))


class OldModel(torch.nn.Module, IOMixIn):
    __version__ = 1

    def __init__(self, n, **kwargs):
        super().__init__()
        self.save_args(locals(), kwargs)
        self.n = n


class NewModel(torch.nn.Module, IOMixIn):
    __version__ = 100000

    def __init__(self, n, **kwargs):
        super().__init__()
        self.save_args(locals(), kwargs)
        self.n = n


def test_model_versioning():
    model = OldModel(1, k=2)
    model.save("OldModel")
    with pytest.warns(UserWarning, match="Incompatible model version detected!"):
        model2 = NewModel.load("OldModel")
    # Model will still load since there are no incompatibilities. Check the properties are reloaded.
    assert isinstance(model2, NewModel)
    assert model2.n == 1
    assert model2._init_args["k"] == 2
    shutil.rmtree("OldModel")


def test_remote_file():
    with RemoteFile(
        "https://github.com/materialsvirtuallab/matgl/raw/main/pretrained_models/MEGNet-MP-2018.6.1-Eform/model.pt",
        cache_location=".",
    ) as s:
        d = torch.load(s, map_location=torch.device("cpu"))
        assert "nblocks" in d["model"]["init_args"]
    try:  # cleanup
        shutil.rmtree("MEGNet-MP-2018.6.1-Eform")
    except FileNotFoundError:
        pass


def test_get_available_pretrained_models():
    model_names = get_available_pretrained_models()
    assert len(model_names) > 1
    assert "M3GNet-MP-2021.2.8-PES" in model_names


def test_load_model():
    # Load model from name.
    model = load_model("M3GNet-MP-2021.2.8-PES")
    assert issubclass(model.__class__, torch.nn.Module)

    # Load model from a full path.

    model = load_model(this_dir / ".." / ".." / "pretrained_models" / "MEGNet-MP-2018.6.1-Eform")
    assert issubclass(model.__class__, torch.nn.Module)
