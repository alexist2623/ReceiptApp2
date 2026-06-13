from .schema import app_field_name, is_dependent_field, is_head_field


def _span_payload(node):
    return {
        "text": node.get("text"),
        "box": node.get("box"),
        "word_indices": node.get("word_indices", []),
        "confidence": node.get("confidence", 1.0),
        "field": node.get("field"),
        "span_id": node.get("span_id"),
    }


def decode_edges_to_items(sample, pair_probs, threshold=0.5, collision_strategy="best_dep"):
    nodes = sample["nodes"]
    pairs = sample["candidate_pairs"].tolist() if hasattr(sample["candidate_pairs"], "tolist") else sample["candidate_pairs"]
    pair_meta = sample.get("pair_meta", [])
    selected = []
    for idx, prob in enumerate(pair_probs):
        prob = float(prob)
        if prob < threshold:
            continue
        meta = dict(pair_meta[idx]) if idx < len(pair_meta) else {}
        head_idx, dep_idx = pairs[idx]
        meta.update({"pair_index": idx, "head_node_id": int(head_idx), "dep_node_id": int(dep_idx), "prob": prob})
        selected.append(meta)

    if collision_strategy == "best_dep":
        best_by_dep = {}
        for edge in selected:
            dep_id = edge["dep_node_id"]
            if dep_id not in best_by_dep or edge["prob"] > best_by_dep[dep_id]["prob"]:
                best_by_dep[dep_id] = edge
        selected = list(best_by_dep.values())

    edges_by_head = {}
    for edge in selected:
        edges_by_head.setdefault(edge["head_node_id"], []).append(edge)

    items = []
    for node in nodes:
        if node.get("node_kind") != "SPAN" or not is_head_field(node.get("field")):
            continue
        item = {
            "item_index": len(items),
            "menu_name": _span_payload(node),
            "price": None,
            "count": None,
            "unit_price": None,
            "discount_price": None,
            "item_subtotal": None,
            "sub_names": [],
            "sub_prices": [],
            "sub_counts": [],
            "sub_unit_prices": [],
            "sub_etc": [],
            "etc": [],
            "rel_g_edges": [],
            "warnings": [],
        }
        for edge in sorted(edges_by_head.get(node["node_id"], []), key=lambda item: item["prob"], reverse=True):
            dep_node = nodes[edge["dep_node_id"]]
            dep_field = dep_node.get("field")
            if not is_dependent_field(dep_field):
                continue
            app_name = app_field_name(dep_field)
            payload = _span_payload(dep_node)
            rel_payload = {
                "head_span_id": node.get("span_id"),
                "dep_span_id": dep_node.get("span_id"),
                "head_text": node.get("text"),
                "dep_text": dep_node.get("text"),
                "dep_field": dep_field,
                "prob": edge["prob"],
            }
            payload["rel_prob"] = edge["prob"]
            if app_name in {"sub_names", "sub_prices", "sub_counts", "sub_unit_prices", "sub_etc", "etc"}:
                item.setdefault(app_name, []).append(payload)
            else:
                if item.get(app_name) is not None:
                    item["warnings"].append(f"Multiple {app_name}; kept highest probability.")
                    continue
                item[app_name] = payload
            item["rel_g_edges"].append(rel_payload)
        if item["price"] is None:
            item["warnings"].append("No MENU_PRICE edge selected.")
        items.append(item)
    return {"items": items, "rel_g_edges": selected}

