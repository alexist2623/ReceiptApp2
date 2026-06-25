from ml.receipt_schema import (
    canonical_output_key,
    canonicalize_field,
    is_document_field,
    is_item_dependent_field,
    is_item_head_field,
    is_payment_field,
    is_summary_field,
)


SUMMARY_RELATION_SLOTS = {
    "SUBTOTAL_NAME": {"SUBTOTAL_PRICE": ("subtotal", "subtotal_price")},
    "TAX_NAME": {
        "TAX_PRICE": ("subtotal", "tax_price"),
        "TAX_RATE": ("subtotal", "tax_rate"),
    },
    "TOTAL_NAME": {"TOTAL_PRICE": ("total", "total_price")},
    "TIP_NAME": {"TIP_PRICE": ("total", "tip_price")},
}


def _span_payload(node):
    return {
        "text": node.get("text"),
        "normalized_text": node.get("normalized_text", node.get("text")),
        "box": node.get("box"),
        "word_indices": node.get("word_indices", []),
        "confidence": node.get("confidence", 1.0),
        "field": canonicalize_field(node.get("field")),
        "raw_field": node.get("raw_field", node.get("field")),
        "span_id": node.get("span_id"),
    }


def _put_once_or_list(container, key, payload):
    current = container.get(key)
    if current is None:
        container[key] = payload
    elif isinstance(current, list):
        current.append(payload)
    else:
        container[key] = [current, payload]


def collect_document_summary_from_nodes(nodes):
    summary = {
        "store": {
            "store_name": None,
            "store_address": None,
            "store_phone": None,
            "store_id": None,
        },
        "subtotal": {
            "subtotal_price": None,
            "tax_name": None,
            "tax_rate": None,
            "tax_price": None,
            "discount_price": None,
            "service_price": None,
        },
        "total": {
            "total_price": None,
            "cash_price": None,
            "change_price": None,
            "card_price": None,
            "tip_price": None,
        },
        "payment": {
            "payment_method": None,
            "payment_card": None,
            "payment_auth_code": None,
            "payment_info": None,
            "approval_code": None,
            "transaction_id": None,
        },
        "document": {
            "receipt_id": None,
            "date": None,
            "time": None,
            "cashier": None,
            "register_id": None,
        },
    }
    for node in nodes:
        if node.get("node_kind", "SPAN") != "SPAN":
            continue
        field = canonicalize_field(node.get("field"))
        payload = _span_payload(node)
        key = canonical_output_key(field)
        if key.startswith("store_"):
            _put_once_or_list(summary["store"], key, payload)
        elif field in {"SUBTOTAL_PRICE", "TAX_NAME", "TAX_RATE", "TAX_PRICE", "DISCOUNT_PRICE", "SERVICE_PRICE"}:
            _put_once_or_list(summary["subtotal"], key, payload)
        elif field in {"TOTAL_PRICE", "CASH_PRICE", "CHANGE_PRICE", "CARD_PRICE", "TIP_PRICE"}:
            _put_once_or_list(summary["total"], key, payload)
        elif is_payment_field(field):
            _put_once_or_list(summary["payment"], key, payload)
        elif is_document_field(field):
            _put_once_or_list(summary["document"], key, payload)
    return summary


def _empty_item(item_index, head_node):
    head_payload = _span_payload(head_node)
    return {
        "item_index": item_index,
        "item_name": head_payload,
        "menu_name": head_payload,
        "name": head_payload,
        "price": None,
        "quantity": None,
        "count": None,
        "unit_price": None,
        "item_code": None,
        "item_sku": None,
        "discount": None,
        "tax_flag": None,
        "options": [],
        "item_etc": [],
        "rel_g_edges": [],
        "warnings": [],
    }


def _item_slot_for_dep(field):
    field = canonicalize_field(field)
    if field == "ITEM_PRICE":
        return "price", False
    if field == "ITEM_QTY":
        return "quantity", False
    if field == "ITEM_UNIT_PRICE":
        return "unit_price", False
    if field == "ITEM_CODE":
        return "item_code", False
    if field == "ITEM_SKU":
        return "item_sku", False
    if field == "ITEM_DISCOUNT":
        return "discount", False
    if field == "ITEM_TAX_FLAG":
        return "tax_flag", False
    if field == "ITEM_OPTION":
        return "options", True
    if field == "ITEM_ETC":
        return "item_etc", True
    return None, False


def _summary_slot_for_edge(head_field, dep_field):
    head_field = canonicalize_field(head_field)
    dep_field = canonicalize_field(dep_field)
    return SUMMARY_RELATION_SLOTS.get(head_field, {}).get(dep_field)


def _select_edges(pair_probs, pairs, pair_meta, threshold, collision_strategy):
    selected = []
    for idx, prob in enumerate(pair_probs):
        prob = float(prob)
        if prob < threshold:
            continue
        meta = dict(pair_meta[idx]) if idx < len(pair_meta) else {}
        head_idx, dep_idx = pairs[idx]
        meta.update(
            {
                "pair_index": idx,
                "head_node_id": int(head_idx),
                "dep_node_id": int(dep_idx),
                "head_field": canonicalize_field(meta.get("head_field")),
                "dep_field": canonicalize_field(meta.get("dep_field")),
                "raw_head_field": meta.get("head_field"),
                "raw_dep_field": meta.get("dep_field"),
                "prob": prob,
            }
        )
        selected.append(meta)

    if collision_strategy != "best_dep":
        return selected

    by_dep = {}
    for edge in selected:
        by_dep.setdefault(edge["dep_node_id"], []).append(edge)
    deduped = []
    for edges in by_dep.values():
        edges = sorted(edges, key=lambda item: item["prob"], reverse=True)
        best = edges[0]
        best["link_margin"] = best["prob"] - edges[1]["prob"] if len(edges) > 1 else None
        best["collision_dropped"] = edges[1:]
        deduped.append(best)
    return deduped


def decode_edges_to_items(sample, pair_probs, threshold=0.84, collision_strategy="best_dep"):
    nodes = sample["nodes"]
    pairs = sample["candidate_pairs"].tolist() if hasattr(sample["candidate_pairs"], "tolist") else sample["candidate_pairs"]
    pair_meta = sample.get("pair_meta", [])
    selected = _select_edges(pair_probs, pairs, pair_meta, threshold, collision_strategy)

    edges_by_head = {}
    for edge in selected:
        edges_by_head.setdefault(edge["head_node_id"], []).append(edge)

    items = []
    ungrouped_spans = []
    summary_relations = []
    for node in nodes:
        if node.get("node_kind") != "SPAN":
            continue
        field = canonicalize_field(node.get("field"))
        if not is_item_head_field(field):
            if is_item_dependent_field(field) or is_summary_field(field) or is_document_field(field) or is_payment_field(field):
                ungrouped_spans.append(_span_payload(node))
            continue
        item = _empty_item(len(items), node)
        for edge in sorted(edges_by_head.get(node["node_id"], []), key=lambda item: item["prob"], reverse=True):
            dep_node = nodes[edge["dep_node_id"]]
            dep_field = canonicalize_field(dep_node.get("field"))
            if not is_item_dependent_field(dep_field):
                continue
            app_name, is_list = _item_slot_for_dep(dep_field)
            if app_name is None:
                continue
            payload = _span_payload(dep_node)
            payload["rel_prob"] = edge["prob"]
            if edge.get("link_margin") is not None:
                payload["link_margin"] = edge["link_margin"]
                if edge["link_margin"] < 0.05:
                    item["warnings"].append(f"Low collision margin for {app_name}: {edge['link_margin']:.4f}")
            rel_payload = {
                "head_span_id": node.get("span_id"),
                "dep_span_id": dep_node.get("span_id"),
                "head_text": node.get("text"),
                "dep_text": dep_node.get("text"),
                "head_field": canonicalize_field(node.get("field")),
                "dep_field": dep_field,
                "raw_dep_field": dep_node.get("raw_field", dep_node.get("field")),
                "prob": edge["prob"],
                "link_margin": edge.get("link_margin"),
            }
            if is_list:
                item.setdefault(app_name, []).append(payload)
            else:
                if item.get(app_name) is not None:
                    item["warnings"].append(f"Multiple {app_name}; kept highest probability.")
                    continue
                item[app_name] = payload
                if app_name == "quantity":
                    item["count"] = payload
            item["rel_g_edges"].append(rel_payload)
        if item["price"] is None:
            item["warnings"].append("No ITEM_PRICE edge selected.")
        items.append(item)

    summary = collect_document_summary_from_nodes(nodes)
    for edge in sorted(selected, key=lambda item: item["prob"], reverse=True):
        head_node = nodes[edge["head_node_id"]]
        dep_node = nodes[edge["dep_node_id"]]
        head_field = canonicalize_field(head_node.get("field"))
        dep_field = canonicalize_field(dep_node.get("field"))
        summary_slot = _summary_slot_for_edge(head_field, dep_field)
        if summary_slot is None:
            continue
        section_name, slot_name = summary_slot
        dep_payload = _span_payload(dep_node)
        dep_payload["rel_prob"] = edge["prob"]
        if edge.get("link_margin") is not None:
            dep_payload["link_margin"] = edge["link_margin"]
        _put_once_or_list(summary[section_name], slot_name, dep_payload)
        summary_relations.append(
            {
                "head_span_id": head_node.get("span_id"),
                "dep_span_id": dep_node.get("span_id"),
                "head_text": head_node.get("text"),
                "dep_text": dep_node.get("text"),
                "head_field": head_field,
                "dep_field": dep_field,
                "prob": edge["prob"],
                "link_margin": edge.get("link_margin"),
            }
        )
    return {
        "schema_version": "receipt_grouped_v2",
        "items": items,
        "store": summary["store"],
        "subtotal": summary["subtotal"],
        "total": summary["total"],
        "payment": summary["payment"],
        "document": summary["document"],
        "ungrouped_spans": ungrouped_spans,
        "summary_relations": summary_relations,
        "rel_g_edges": selected,
        "warnings": [],
    }
