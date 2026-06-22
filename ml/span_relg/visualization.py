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


def _merge_boxes(boxes):
    boxes = [box for box in boxes if isinstance(box, list) and len(box) == 4]
    if not boxes:
        return None
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def _node_sort_key(node):
    value = node.get("first_word_idx")
    if value is not None:
        return int(value)
    indices = node.get("word_indices") or []
    return int(min(indices)) if indices else 10**9


def _join_node_text(nodes):
    return " ".join(
        str(node.get("text") or "").strip()
        for node in sorted(nodes, key=_node_sort_key)
        if str(node.get("text") or "").strip()
    )


def _short_text(text, limit=44):
    text = str(text or "")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _edge_status_for_group(edges, head_nodes, dep_nodes):
    head_ids = {node.get("node_id") for node in head_nodes}
    dep_ids = {node.get("node_id") for node in dep_nodes}
    matches = [
        edge
        for edge in edges
        if edge.get("head_node_id") in head_ids and edge.get("dep_node_id") in dep_ids
    ]
    if not matches:
        return None, "grouped"
    matches = sorted(matches, key=lambda edge: float(edge.get("prob", 0.0)), reverse=True)
    edge = matches[0]
    status = edge.get("status")
    if status == "missed":
        label = "missed"
    elif edge.get("correct") is False:
        label = "wrong"
    elif edge.get("correct") is True:
        label = "ok"
    else:
        label = "selected"
    return edge, label


def _grouped_mapping_rows_from_sample(sample, edges):
    nodes = [node for node in sample.get("nodes", []) if node.get("node_kind") == "SPAN"]
    nodes_by_id = {node.get("node_id"): node for node in nodes}
    grouped = {}
    for node in nodes:
        group_key = node.get("group_key")
        if group_key:
            grouped.setdefault(group_key, []).append(node)

    rows = []
    seen_pairs = set()
    for group_key, group_nodes in sorted(grouped.items(), key=lambda item: min(_node_sort_key(node) for node in item[1])):
        item_names = [node for node in group_nodes if node.get("field") == "ITEM_NAME"]
        item_options = [node for node in group_nodes if node.get("field") == "ITEM_OPTION"]
        item_prices = [node for node in group_nodes if node.get("field") == "ITEM_PRICE"]
        item_qty = [node for node in group_nodes if node.get("field") == "ITEM_QTY"]
        if item_names:
            head_nodes = item_names + item_options
            edge, status = _edge_status_for_group(edges, item_names, item_prices)
            rows.append(
                {
                    "kind": "ITEM",
                    "group_key": group_key,
                    "head_text": _join_node_text(head_nodes),
                    "tail_text": _join_node_text(item_prices) or "(no ITEM_PRICE span in group)",
                    "meta_text": "qty=" + _join_node_text(item_qty) if item_qty else "",
                    "head_box": _merge_boxes([node.get("box") for node in head_nodes]),
                    "tail_box": _merge_boxes([node.get("box") for node in item_prices]),
                    "status": status,
                    "prob": edge.get("prob") if edge else None,
                }
            )
            for head in item_names:
                for dep in item_prices:
                    seen_pairs.add((head.get("node_id"), dep.get("node_id")))
            continue

        summary_names = [
            node
            for node in group_nodes
            if node.get("field")
            in {
                "SUBTOTAL_NAME",
                "TAX_NAME",
                "TOTAL_NAME",
                "TIP_NAME",
                "DISCOUNT_NAME",
                "SERVICE_NAME",
            }
        ]
        summary_amounts = [
            node
            for node in group_nodes
            if node.get("field")
            in {
                "SUBTOTAL_PRICE",
                "TAX_PRICE",
                "TAX_RATE",
                "TOTAL_PRICE",
                "TIP_PRICE",
                "DISCOUNT_PRICE",
                "SERVICE_PRICE",
            }
        ]
        if summary_names and summary_amounts:
            edge, status = _edge_status_for_group(edges, summary_names, summary_amounts)
            rows.append(
                {
                    "kind": summary_names[0].get("field", "SUMMARY").replace("_NAME", ""),
                    "group_key": group_key,
                    "head_text": _join_node_text(summary_names),
                    "tail_text": _join_node_text(summary_amounts),
                    "meta_text": ",".join(sorted({node.get("field") for node in summary_amounts})),
                    "head_box": _merge_boxes([node.get("box") for node in summary_names]),
                    "tail_box": _merge_boxes([node.get("box") for node in summary_amounts]),
                    "status": status,
                    "prob": edge.get("prob") if edge else None,
                }
            )
            for head in summary_names:
                for dep in summary_amounts:
                    seen_pairs.add((head.get("node_id"), dep.get("node_id")))

    # Also show selected predicted summary/tax edges that were not represented
    # by a complete relation group. This catches useful failure cases such as
    # TAX_NAME being predicted correctly while its amount/rate linkage is wrong
    # or the gold group assignment is absent.
    summary_heads = {
        "SUBTOTAL_NAME",
        "TAX_NAME",
        "TOTAL_NAME",
        "TIP_NAME",
        "DISCOUNT_NAME",
        "SERVICE_NAME",
        "PAYMENT_METHOD",
    }
    summary_deps = {
        "SUBTOTAL_PRICE",
        "TAX_PRICE",
        "TAX_RATE",
        "TOTAL_PRICE",
        "TIP_PRICE",
        "DISCOUNT_PRICE",
        "SERVICE_PRICE",
        "PAYMENT_CARD",
        "PAYMENT_INFO",
    }
    for edge in sorted(edges, key=lambda item: float(item.get("prob", 0.0)), reverse=True):
        if edge.get("status") == "missed" or edge.get("pred") == 0:
            continue
        head = nodes_by_id.get(edge.get("head_node_id"))
        dep = nodes_by_id.get(edge.get("dep_node_id"))
        if not head or not dep:
            continue
        head_field = head.get("field")
        dep_field = dep.get("field")
        pair_key = (head.get("node_id"), dep.get("node_id"))
        if pair_key in seen_pairs or head_field not in summary_heads or dep_field not in summary_deps:
            continue
        status = edge.get("status")
        if not status:
            status = "wrong" if edge.get("correct") is False else "selected"
        rows.append(
            {
                "kind": head_field.replace("_NAME", ""),
                "group_key": edge.get("head_group_key") or edge.get("dep_group_key") or "selected",
                "head_text": head.get("text"),
                "tail_text": dep.get("text"),
                "meta_text": dep_field,
                "head_box": head.get("box"),
                "tail_box": dep.get("box"),
                "status": status,
                "prob": edge.get("prob"),
            }
        )
        seen_pairs.add(pair_key)
    return rows


def draw_grouped_relation_mapping_overlay(image, sample, edges, out_path, title=None):
    """Draw item/summary mappings grouped by relation group_key.

    This view is intended for labeled custom evaluation overlays. It avoids
    showing multi-line ITEM_NAME continuations as separate "missing price" rows
    when the labeled relation group says they belong to the same item.
    """
    rows = _grouped_mapping_rows_from_sample(sample, edges)
    if not rows:
        return draw_span_relg_overlay(image, sample, edges, out_path, title=title)

    base = image.copy().convert("RGB")
    panel_width = max(900, int(base.width * 0.36))
    output = Image.new("RGB", (base.width + panel_width, base.height), (248, 250, 252))
    output.paste(base, (0, 0))
    draw = ImageDraw.Draw(output)

    title_font = _load_font(34, bold=True)
    header_font = _load_font(25, bold=True)
    text_font = _load_font(22)
    small_font = _load_font(18)
    tag_font = _load_font(20, bold=True)

    palette = [
        (37, 99, 235),
        (5, 150, 105),
        (217, 119, 6),
        (147, 51, 234),
        (8, 145, 178),
        (190, 24, 93),
        (101, 163, 13),
        (100, 116, 139),
    ]
    status_colors = {
        "ok": (22, 163, 74),
        "selected": (22, 163, 74),
        "grouped": (37, 99, 235),
        "missed": (37, 99, 235),
        "wrong": (220, 38, 38),
    }

    if title:
        _draw_text_box(draw, (16, 16), title, (17, 24, 39), title_font, background=(255, 255, 255), padding=8)

    panel_x = base.width
    draw.rectangle([panel_x, 0, output.width - 1, output.height - 1], fill=(248, 250, 252), outline=(203, 213, 225))
    draw.text((panel_x + 28, 26), "Grouped Mapping", fill=(15, 23, 42), font=title_font)
    draw.text(
        (panel_x + 28, 76),
        "ITEM_NAME B/I continuations are grouped by relation group.",
        fill=(71, 85, 105),
        font=small_font,
    )
    y = 122

    for row_index, row in enumerate(rows, start=1):
        color = palette[(row_index - 1) % len(palette)]
        status_color = status_colors.get(row.get("status"), color)
        head_box = row.get("head_box")
        tail_box = row.get("tail_box")
        if head_box:
            _draw_rect(draw, head_box, color, width=6)
            _draw_text_box(
                draw,
                (head_box[0], max(0, head_box[1] - 32)),
                f"#{row_index}",
                color,
                tag_font,
                background=(255, 255, 255),
                padding=5,
            )
        if tail_box:
            _draw_rect(draw, tail_box, status_color, width=6)
        if head_box and tail_box:
            _draw_arrow(draw, _center(head_box), _center(tail_box), status_color, width=5)

        if y > base.height - 110:
            draw.text((panel_x + 28, y), "... more grouped mappings", fill=(100, 116, 139), font=small_font)
            break
        draw.rectangle([panel_x + 24, y - 8, output.width - 24, y + 112], fill=(255, 255, 255), outline=(226, 232, 240))
        draw.rectangle([panel_x + 24, y - 8, panel_x + 36, y + 112], fill=color)
        draw.text((panel_x + 48, y), f"#{row_index} {row['kind']}", fill=color, font=header_font)
        draw.text((panel_x + 112, y), _short_text(row.get("head_text"), 46), fill=(15, 23, 42), font=text_font)
        tail = _short_text(row.get("tail_text"), 34)
        prob = row.get("prob")
        meta = row.get("meta_text") or ""
        status = row.get("status") or ""
        suffix = f"{status}"
        if prob is not None:
            suffix += f" p={float(prob):.2f}"
        if meta:
            suffix += f" | {meta}"
        draw.text((panel_x + 112, y + 36), f"-> {tail}", fill=status_color, font=text_font)
        draw.text((panel_x + 112, y + 68), _short_text(suffix, 64), fill=(100, 116, 139), font=small_font)
        y += 132

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(out_path)
    return out_path


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
