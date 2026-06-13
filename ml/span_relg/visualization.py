from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont


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


def _load_font(size=24, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_size(draw, text, font):
    try:
        box = draw.textbbox((0, 0), text, font=font)
        return box[2] - box[0], box[3] - box[1]
    except AttributeError:
        return draw.textsize(text, font=font)


def _draw_text_box(draw, xy, text, fill, font, background=(255, 255, 255), padding=4):
    x, y = xy
    width, height = _text_size(draw, text, font)
    draw.rectangle([x, y, x + width + padding * 2, y + height + padding * 2], fill=background, outline=fill)
    draw.text((x + padding, y + padding), text, fill=fill, font=font)


def _draw_arrow(draw, start, end, color, width=4):
    draw.line([start[0], start[1], end[0], end[1]], fill=color, width=width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux, uy = dx / length, dy / length
    size = max(10, width * 4)
    left = (end[0] - ux * size - uy * size * 0.55, end[1] - uy * size + ux * size * 0.55)
    right = (end[0] - ux * size + uy * size * 0.55, end[1] - uy * size - ux * size * 0.55)
    try:
        draw.polygon([end, left, right], fill=color)
    except TypeError:
        pass


def _field_text(value):
    return value.get("text", "") if isinstance(value, dict) else ""


def _field_box(value):
    box = value.get("box") if isinstance(value, dict) else None
    return box if isinstance(box, list) and len(box) == 4 else None


def _field_prob(value):
    if isinstance(value, dict) and value.get("rel_prob") is not None:
        try:
            return float(value["rel_prob"])
        except (TypeError, ValueError):
            return None
    return None


def draw_user_item_mapping_overlay(image, items, out_path, title=None):
    """Draw a clean user-facing item -> price overlay.

    Unlike the graph-debug overlay, this view shows only decoded item mappings:
    item boxes, price boxes, optional quantity/unit-price boxes, and a side table.
    """
    base = image.copy().convert("RGB")
    panel_width = max(980, int(base.width * 0.36))
    output = Image.new("RGB", (base.width + panel_width, base.height), (248, 250, 252))
    output.paste(base, (0, 0))
    draw = ImageDraw.Draw(output)

    title_font = _load_font(36, bold=True)
    header_font = _load_font(27, bold=True)
    text_font = _load_font(24)
    small_font = _load_font(20)
    tag_font = _load_font(22, bold=True)

    palette = [
        (37, 99, 235),
        (5, 150, 105),
        (217, 119, 6),
        (147, 51, 234),
        (220, 38, 38),
        (8, 145, 178),
        (101, 163, 13),
        (190, 24, 93),
    ]
    price_color = (22, 163, 74)
    no_price_color = (220, 38, 38)
    helper_color = (107, 114, 128)

    if title:
        _draw_text_box(draw, (16, 16), title, (17, 24, 39), title_font, background=(255, 255, 255), padding=8)

    panel_x = base.width
    draw.rectangle([panel_x, 0, output.width - 1, output.height - 1], fill=(248, 250, 252), outline=(203, 213, 225))
    draw.text((panel_x + 28, 26), "Item -> Price Mapping", fill=(15, 23, 42), font=title_font)
    draw.text(
        (panel_x + 28, 78),
        "Blue box = item span, green box = linked price. Red row = no price.",
        fill=(71, 85, 105),
        font=small_font,
    )
    y = 126

    for item in items:
        if not isinstance(item, dict):
            continue
        item_index = item.get("item_index", 0)
        color = palette[int(item_index or 0) % len(palette)]
        menu = item.get("item_name") or item.get("menu_name") or item.get("name")
        price = item.get("price") or item.get("menu_price")
        count = item.get("quantity") or item.get("count")
        unit_price = item.get("unit_price")
        warnings = item.get("warnings") or []
        menu_text = _field_text(menu)
        price_text = _field_text(price)
        prob = _field_prob(price)

        menu_box = _field_box(menu)
        price_box = _field_box(price)
        if menu_box:
            _draw_rect(draw, menu_box, color, width=6)
            _draw_text_box(
                draw,
                (menu_box[0], max(0, menu_box[1] - 34)),
                f"#{item_index} ITEM",
                color,
                tag_font,
                background=(255, 255, 255),
                padding=5,
            )
        if price_box:
            _draw_rect(draw, price_box, price_color, width=6)
            label = f"#{item_index} PRICE {price_text}"
            if prob is not None:
                label += f" {prob:.2f}"
            _draw_text_box(
                draw,
                (price_box[0], max(0, price_box[1] - 34)),
                label,
                price_color,
                tag_font,
                background=(255, 255, 255),
                padding=5,
            )
        if menu_box and price_box:
            _draw_arrow(draw, _center(menu_box), _center(price_box), price_color, width=5)
        elif menu_box:
            _draw_text_box(
                draw,
                (menu_box[0], min(base.height - 34, menu_box[3] + 4)),
                f"#{item_index} NO PRICE",
                no_price_color,
                tag_font,
                background=(255, 245, 245),
                padding=5,
            )

        for helper_name, helper_value in (("CNT", count), ("UNIT", unit_price)):
            helper_box = _field_box(helper_value)
            if helper_box:
                _draw_rect(draw, helper_box, helper_color, width=3)
                _draw_text_box(
                    draw,
                    (helper_box[0], max(0, helper_box[1] - 26)),
                    f"#{item_index} {helper_name}",
                    helper_color,
                    small_font,
                    background=(255, 255, 255),
                    padding=4,
                )

        row_color = no_price_color if not price_text else (15, 23, 42)
        draw.rectangle([panel_x + 24, y - 8, output.width - 24, y + 104], fill=(255, 255, 255), outline=(226, 232, 240))
        draw.rectangle([panel_x + 24, y - 8, panel_x + 36, y + 104], fill=color)
        draw.text((panel_x + 48, y), f"#{item_index}", fill=color, font=header_font)
        summary = f"{menu_text or '(no menu)'} -> {price_text or '(no price)'}"
        wrapped = textwrap.wrap(summary, width=42)[:2]
        for line_idx, line in enumerate(wrapped):
            draw.text((panel_x + 112, y + line_idx * 28), line, fill=row_color, font=text_font)
        meta = []
        if prob is not None:
            meta.append(f"p={prob:.3f}")
        if _field_text(count):
            meta.append(f"cnt={_field_text(count)}")
        if _field_text(unit_price):
            meta.append(f"unit={_field_text(unit_price)}")
        if warnings:
            meta.append("; ".join(warnings[:2]))
        if meta:
            draw.text((panel_x + 112, y + 62), " | ".join(meta)[:70], fill=(100, 116, 139), font=small_font)
        y += 124
        if y > base.height - 90:
            draw.text((panel_x + 28, y), "... more items in grouped JSON", fill=(100, 116, 139), font=small_font)
            break

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(out_path)
    return out_path


def draw_span_relg_overlay(image, sample, edges, out_path, title=None):
    output = image.copy().convert("RGB")
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    nodes = sample["nodes"]
    for node in nodes:
        if node.get("node_kind") != "SPAN":
            continue
        color = (30, 120, 210) if node.get("field") in {"ITEM_NAME", "MENU_NM"} else (80, 80, 80)
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
