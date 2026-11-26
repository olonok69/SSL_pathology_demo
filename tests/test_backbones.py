import torch
from PIL import Image

from ssl_demo.backbones import (
    PHASE1_BACKBONES,
    backbone_info,
    get_backbone,
    list_backbones,
    make_preprocess,
)


def test_registry_lists_the_phase1_models():
    assert list_backbones() == PHASE1_BACKBONES
    assert backbone_info("random_resnet50").embed_dim == 2048
    assert backbone_info("sup_resnet50_in1k").embed_dim == 2048
    assert backbone_info("dinov2_vits14").embed_dim == 384
    assert backbone_info("dinov2_vits14").kind == "cls+patchmean"
    try:
        backbone_info("phikon")
    except KeyError:
        pass
    else:
        raise AssertionError("phikon is not in this slice")


def test_random_resnet_shape_and_normalisation():
    model, preprocess, embed_dim = get_backbone("random_resnet50")
    assert embed_dim == 2048
    assert model.training is False
    batch = torch.rand(1, 3, 224, 224)
    with torch.inference_mode():
        output = model(batch)
    assert output.shape == (1, 2048)

    white = Image.new("RGB", (300, 280), color=(255, 255, 255))
    tensor = preprocess(white)
    assert tensor.shape == (3, 224, 224)
    info = backbone_info("random_resnet50")
    expected = (1.0 - info.mean[0]) / info.std[0]
    assert torch.allclose(tensor[0], torch.full_like(tensor[0], expected), atol=1e-5)


def test_preprocess_uses_the_given_stats():
    preprocess = make_preprocess((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 16)
    image = Image.new("RGB", (16, 16), color=(255, 0, 0))
    tensor = preprocess(image)
    assert tensor.shape == (3, 16, 16)
    assert torch.allclose(tensor[0], torch.ones(16, 16), atol=1e-5)
    assert torch.allclose(tensor[1], torch.zeros(16, 16), atol=1e-5)
