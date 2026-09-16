import pytest
import torch

from adarep_kd.models import CIFARBasicBlock, resnet8, resnet32


BATCH_SIZE = 4
NUM_CLASSES = 100

EXPECTED_FEATURE_SHAPES = {
    "stage1": (BATCH_SIZE, 16, 32, 32),
    "stage2": (BATCH_SIZE, 32, 16, 16),
    "stage3": (BATCH_SIZE, 64, 8, 8),
}


@pytest.fixture
def input_batch():
    return torch.randn(BATCH_SIZE, 3, 32, 32)


@pytest.mark.parametrize(
    "model_factory, expected_blocks",
    [
        (resnet32, 15),
        (resnet8, 3),
    ],
)
def test_resnet_depth(model_factory, expected_blocks):
    model = model_factory()

    residual_blocks = sum(
        1
        for module in model.modules()
        if isinstance(module, CIFARBasicBlock)
    )

    assert residual_blocks == expected_blocks


@pytest.mark.parametrize("model_factory", [resnet32, resnet8])
def test_logits_shape(model_factory, input_batch):
    model = model_factory()

    logits, _ = model(input_batch)

    assert logits.shape == (BATCH_SIZE, NUM_CLASSES)


@pytest.mark.parametrize("model_factory", [resnet32, resnet8])
def test_feature_keys(model_factory, input_batch):
    model = model_factory()

    _, features = model(input_batch)

    assert set(features.keys()) == {
        "stage1",
        "stage2",
        "stage3",
    }


@pytest.mark.parametrize("model_factory", [resnet32, resnet8])
def test_feature_shapes(model_factory, input_batch):
    model = model_factory()

    _, features = model(input_batch)

    for stage, expected_shape in EXPECTED_FEATURE_SHAPES.items():
        assert features[stage].shape == expected_shape


@pytest.mark.parametrize("model_factory", [resnet32, resnet8])
def test_batch_dimension_is_preserved(model_factory):
    model = model_factory()

    for batch_size in [1, 2, 8]:
        x = torch.randn(batch_size, 3, 32, 32)

        logits, features = model(x)

        assert logits.shape[0] == batch_size

        for feature in features.values():
            assert feature.shape[0] == batch_size


@pytest.mark.parametrize("model_factory", [resnet32, resnet8])
def test_forward_returns_tensors(model_factory, input_batch):
    model = model_factory()

    logits, features = model(input_batch)

    assert isinstance(logits, torch.Tensor)

    for feature in features.values():
        assert isinstance(feature, torch.Tensor)


def test_teacher_and_student_have_matching_feature_shapes(input_batch):
    teacher = resnet32()
    student = resnet8()

    _, teacher_features = teacher(input_batch)
    _, student_features = student(input_batch)

    assert teacher_features.keys() == student_features.keys()

    for stage in teacher_features:
        assert teacher_features[stage].shape == student_features[stage].shape