from ml.angle_geometry import angle_deg_from_quad, box_to_quad, clamp_box, quad_to_axis_aligned_box, rotate_points


def test_rotated_box_keeps_valid_axis_aligned_bbox():
    source_box = [20, 20, 80, 40]
    quad = box_to_quad(source_box)
    rotated = rotate_points(quad, width=120, height=80, angle_deg=12)
    box = clamp_box(quad_to_axis_aligned_box(rotated), 136, 104)
    assert box is not None
    assert box[2] > box[0]
    assert box[3] > box[1]


def test_rotated_quad_has_nonzero_angle():
    quad = box_to_quad([20, 20, 80, 40])
    rotated = rotate_points(quad, width=120, height=80, angle_deg=12)
    angle = angle_deg_from_quad(rotated)
    assert angle is not None
    assert abs(angle) > 1.0
