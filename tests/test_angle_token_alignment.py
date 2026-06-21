import torch

from ml.angle_geometry import ANGLE_FEATURE_DIM, align_angle_features_to_tokens


class FakeEncoding(dict):
    def __init__(self):
        super().__init__()
        self["input_ids"] = torch.tensor([[0, 10, 11, 12, 2, 1]])
        self["attention_mask"] = torch.tensor([[1, 1, 1, 1, 1, 0]])
        self._word_ids = [None, 0, 0, 1, None, None]

    def word_ids(self, batch_index=0):
        assert batch_index == 0
        return self._word_ids


def test_align_angle_features_to_all_subwords():
    encoding = FakeEncoding()
    word_features = [[1.0] * ANGLE_FEATURE_DIM, [2.0] * ANGLE_FEATURE_DIM]
    aligned = align_angle_features_to_tokens(encoding, word_features, batch_index=0)
    assert aligned.shape == (6, ANGLE_FEATURE_DIM)
    assert aligned[0].sum().item() == 0.0
    assert aligned[1].tolist() == word_features[0]
    assert aligned[2].tolist() == word_features[0]
    assert aligned[3].tolist() == word_features[1]
    assert aligned[5].sum().item() == 0.0


def test_align_angle_features_first_subword_only():
    encoding = FakeEncoding()
    word_features = [[1.0] * ANGLE_FEATURE_DIM, [2.0] * ANGLE_FEATURE_DIM]
    aligned = align_angle_features_to_tokens(
        encoding,
        word_features,
        batch_index=0,
        first_subword_only=True,
    )
    assert aligned[1].tolist() == word_features[0]
    assert aligned[2].sum().item() == 0.0
    assert aligned[3].tolist() == word_features[1]
