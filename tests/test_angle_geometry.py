import math

from ml.angle_geometry import (
    ANGLE_FEATURE_DIM,
    angle_deg_from_quad,
    build_word_angle_features,
    normalize_angle_deg,
    parse_quad,
    quad_to_axis_aligned_box,
)


def test_parse_quad_and_box_from_corner_points():
    payload = {"cornerPoints": [[10, 20], [110, 25], [108, 50], [8, 45]]}
    quad = parse_quad(payload)
    assert quad == [(10.0, 20.0), (110.0, 25.0), (108.0, 50.0), (8.0, 45.0)]
    assert quad_to_axis_aligned_box(payload) == [8, 20, 110, 50]


def test_angle_deg_from_slightly_rotated_quad():
    angle = angle_deg_from_quad([[0, 0], [100, 10], [100, 30], [0, 20]])
    assert angle is not None
    assert math.isclose(angle, 5.7106, abs_tol=0.1)


def test_normalize_angle_wraps_to_signed_range():
    assert normalize_angle_deg(190) == -170
    assert normalize_angle_deg(-190) == 170


def test_missing_angle_features_are_zero():
    features, debug = build_word_angle_features({"text": "abc", "box": [0, 0, 10, 10]}, image_width=100, image_height=100)
    assert len(features) == ANGLE_FEATURE_DIM
    assert features == [0.0] * ANGLE_FEATURE_DIM
    assert debug["has_angle"] is False


def test_explicit_angle_features_include_has_angle():
    features, debug = build_word_angle_features(
        {"text": "abc", "box": [0, 0, 20, 10], "angleDeg": 12.0},
        image_width=100,
        image_height=100,
    )
    assert len(features) == ANGLE_FEATURE_DIM
    assert features[4] == 1.0
    assert debug["angle_deg"] == 12.0
