from pathlib import Path

from PIL import ImageDraw, ImageFont


def _center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _draw_rect(draw, box, color, width=2):
    try:
        draw.rectangle(box, outline=color, width=width)
    except TypeError:
        for offset in range(width):
            draw.rectangle(
                [box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset],
                outline=color,
            )


def draw_span_relg_overlay(image, sample, edges, out_path, title=None):
    output = image.copy().convert("RGB")
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    nodes = sample["nodes"]
    for node in nodes:
        if node.get("node_kind") != "SPAN":
            continue
        color = (30, 120, 210) if node.get("field") == "MENU_NM" else (80, 80, 80)
        _draw_rect(draw, node["box"], color, width=2)
        draw.text((node["box"][0], max(0, node["box"][1] - 12)), f"{node['field']} {node['text'][:18]}", fill=color, font=font)

    for edge in edges:
        head = nodes[edge["head_node_id"]]
        dep = nodes[edge["dep_node_id"]]
        status = edge.get("status")
        if edge.get("hard_negative", False):
            color = (165, 40, 190)
        elif status == "missed":
            color = (45, 95, 220)
        elif edge.get("correct", False) or status == "correct":
            color = (40, 170, 80)
        else:
            color = (210, 60, 50)
        x0, y0 = _center(head["box"])
        x1, y1 = _center(dep["box"])
        width = 4 if edge.get("hard_negative", False) else 3
        draw.line([x0, y0, x1, y1], fill=color, width=width)
        try:
            draw.ellipse([x1 - 3, y1 - 3, x1 + 3, y1 + 3], fill=color)
        except TypeError:
            pass
        label = f"{edge.get('dep_field', '')} {edge.get('prob', 0):.2f}".strip()
        draw.text(((x0 + x1) / 2, (y0 + y1) / 2), label, fill=color, font=font)
    if title:
        draw.text((8, 8), title, fill=(0, 0, 0), font=font)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(out_path)
    return out_path
