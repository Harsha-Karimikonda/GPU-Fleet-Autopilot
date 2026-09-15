#!/usr/bin/env python3
"""
Exports a trained XGBoost model into the neutral JSON decision tree format
conforming to schema/model_v1.json for pure-Rust runtime inference.
"""

import json
import sys
from typing import Any, Dict


def export_xgboost_to_neutral_json(booster, feature_names, output_path: str, base_score: float = 0.5):
    dump = booster.get_dump(dump_format="json")
    trees = []

    for tree_idx, tree_raw in enumerate(dump):
        tree_obj = json.loads(tree_raw)
        nodes = []

        def parse_node(node_dict: Dict[str, Any]):
            node_id = node_dict["nodeid"]
            if "leaf" in node_dict:
                nodes.append({
                    "id": node_id,
                    "leaf_value": float(node_dict["leaf"])
                })
            else:
                feat_name = node_dict["split"]
                try:
                    split_idx = feature_names.index(feat_name)
                except ValueError:
                    # In case feature name is 'f0', 'f1', etc.
                    split_idx = int(feat_name.lstrip("f"))

                nodes.append({
                    "id": node_id,
                    "split_feature": split_idx,
                    "threshold": float(node_dict["split_condition"]),
                    "yes": node_dict["yes"],
                    "no": node_dict["no"],
                    "missing": node_dict.get("missing", node_dict["yes"]),
                })

                for child in node_dict.get("children", []):
                    parse_node(child)

        parse_node(tree_obj)
        trees.append({"nodes": sorted(nodes, key=lambda n: n["id"])})

    export_payload = {
        "schema_version": "1.0",
        "model_type": "xgboost",
        "base_score": base_score,
        "objective": "binary:logistic",
        "num_trees": len(trees),
        "features": feature_names,
        "trees": trees,
    }

    with open(output_path, "w") as f:
        json.dump(export_payload, f, indent=2)

    print(f"Exported {len(trees)} trees to {output_path}")


if __name__ == "__main__":
    print("XGBoost export module loaded.")

