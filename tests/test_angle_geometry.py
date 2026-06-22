import math

from ml.angle_geometry import (
    ANGLE_FEATURE_DIM,
    ANGLE_QUAD_FEATURE_DIM,
    angle_feature_dim_for_mode,
    angle_deg_from_quad,
    build_word_angle_features,
    normalize_angle_deg,
    parse_quad,
    relative_quad_features,
    rotate_points,
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
    assert normalize_angle_deg(91) == -89
    assert normalize_angle_deg(-91) == 89
    assert normalize_angle_deg(180) == 0


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


def test_angle_quad_mode_has_expected_dim():
    features, debug = build_word_angle_features(
        {"text": "abc", "quad": [0, 0, 20, 4, 18, 14, -2, 10], "angle_deg": 10.0},
        box=[0, 0, 20, 14],
        image_width=100,
        image_height=100,
        mode="angle_quad",
    )
    assert len(features) == ANGLE_QUAD_FEATURE_DIM
    assert angle_feature_dim_for_mode("angle_quad") == ANGLE_QUAD_FEATURE_DIM
    assert debug["has_quad"] is True


def test_relative_quad_features_are_eight_values():
    features = relative_quad_features([0, 0, 20, 4, 18, 14, -2, 10], [0, 0, 20, 14])
    assert len(features) == 8


def test_rotate_points_changes_canvas_with_expand():
    points = rotate_points([(0, 0), (100, 0), (100, 50), (0, 50)], 100, 50, 10)
    assert len(points) == 4
    assert min(x for x, _ in points) >= -1e-6
