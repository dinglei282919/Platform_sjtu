"""SDG-HAZOP and LOPA adapter without any web/UI state."""

from __future__ import annotations

import math
from typing import Any


DEFAULT_NODES = [
    {"id": "R1", "name": "冷却水故障", "type": "R", "probability": 0.018},
    {"id": "R2", "name": "搅拌器故障", "type": "R", "probability": 0.008},
    {"id": "T1", "name": "反应器温度", "type": "P", "probability": 0.0},
    {"id": "P1", "name": "反应器压力", "type": "P", "probability": 0.0},
    {"id": "C1", "name": "反应器爆炸", "type": "C", "probability": 0.0},
    {"id": "C2", "name": "安全阀起跳", "type": "C", "probability": 0.0},
]
DEFAULT_EDGES = [
    {"source": "R1", "target": "T1", "type": "+", "probability": 0.85},
    {"source": "R2", "target": "T1", "type": "+", "probability": 0.70},
    {"source": "T1", "target": "P1", "type": "+", "probability": 0.90},
    {"source": "P1", "target": "C1", "type": "+", "probability": 0.65},
    {"source": "P1", "target": "C2", "type": "+", "probability": 0.40},
]


def config() -> dict[str, Any]:
    """Return the CS-compatible editor defaults and fuzzy term projections."""
    from sdg_hazop import FuzzyExpertEvaluator

    evaluator = FuzzyExpertEvaluator()
    fuzzy_terms = [
        {"label": term, "probability": evaluator.evaluate_term(term, is_frequency=True)}
        for term in evaluator.FUZZY_TERMS
    ]
    return {
        "fuzzy_terms": fuzzy_terms,
        "node_defaults": {"id": "R1", "name": "冷却水故障", "type": "R", "probability": 0.018, "fuzzy_term": "中等"},
        "edge_defaults": {"type": "+", "probability": 0.85},
        "node_types": [{"value": "R", "label": "原因 (R)"}, {"value": "P", "label": "参数 (P)"}, {"value": "C", "label": "后果 (C)"}],
        "edge_types": [{"value": "+", "label": "增量 (+)"}, {"value": "-", "label": "减量 (-)"}],
    }


def analyze(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    from sdg_hazop import EdgeType, NodeType, ProbabilisticSDG, RiskMatrix, SDGEdge, SDGNode

    if len(nodes) < 2:
        raise ValueError("至少需要两个节点")
    model = ProbabilisticSDG()
    seen: set[str] = set()
    for node in nodes:
        node_id = node["id"].strip()
        node_name = str(node.get("name", "")).strip()
        node_type = node["type"]
        if not node_id or not node_name or node_id in seen or node_type not in {"R", "P", "C"}:
            raise ValueError("节点 ID 和名称不能为空，节点 ID 必须唯一，节点类型必须为 R、P 或 C")
        seen.add(node_id)
        probability = float(node.get("probability", 0.0))
        if not math.isfinite(probability) or probability < 0:
            raise ValueError(f"节点 {node_id} 的概率/频率必须是有限且非负数")
        model.add_node(
            SDGNode(
                node_id,
                node_name,
                NodeType.CAUSE if node_type == "R" else NodeType.PARAMETER if node_type == "P" else NodeType.CONSEQUENCE,
                probability if node_type == "R" else None,
            )
        )
    for edge in edges:
        source, target = str(edge["source"]).strip(), str(edge["target"]).strip()
        if source not in seen or target not in seen:
            raise ValueError(f"边 {source}→{target} 引用了不存在的节点")
        if source == target:
            raise ValueError("边的源节点和目标节点不能相同")
        edge_type = edge.get("type", "+")
        if edge_type not in {"+", "-"}:
            raise ValueError("边的影响类型必须为 + 或 -")
        probability = float(edge["probability"])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(f"边 {source}→{target} 的条件概率必须在 0～1 之间")
        model.add_edge(SDGEdge(source, target, EdgeType.INCREMENT if edge_type == "+" else EdgeType.DECREMENT, probability))

    causes = [node["id"] for node in nodes if node["type"] == "R"]
    consequences = [node["id"] for node in nodes if node["type"] == "C"]
    if not causes or not consequences:
        raise ValueError("模型至少需要一个原因R节点和一个后果C节点")

    forward_paths: list[dict[str, Any]] = []
    for cause in causes:
        for path in model.forward_reasoning(cause):
            probability, steps = model.calculate_path_probability(path)
            forward_paths.append({"cause": cause, "path": path, "probability": probability, "steps": steps})

    consequence_results: list[dict[str, Any]] = []
    backward_paths: list[dict[str, Any]] = []
    recommendations: list[dict[str, Any]] = []
    for consequence in consequences:
        paths = model.backward_reasoning(consequence)
        backward_paths.append({"consequence": consequence, "paths": [list(reversed(path)) for path in paths]})
        path_probabilities: list[float] = []
        path_details: list[dict[str, Any]] = []
        for path in paths:
            ordered_path = list(reversed(path))
            probability, steps = model.calculate_path_probability(ordered_path)
            path_probabilities.append(probability)
            path_details.append({"path": ordered_path, "probability": probability, "steps": steps})
        if len(path_probabilities) == 1:
            total = path_probabilities[0]
            aggregation_steps: list[str] = []
        elif path_probabilities:
            total, aggregation_steps = model.calculate_or_probability(path_probabilities)
        else:
            total, aggregation_steps = 0.0, []

        name = next(node["name"] for node in nodes if node["id"] == consequence)
        probability_level, probability_desc = RiskMatrix.get_prob_level(total)
        severity_level, severity_desc = RiskMatrix.get_sev_level(name)
        risk_level, action = RiskMatrix.get_risk(probability_level, severity_level)
        residual = total * 0.1 * 0.01
        tolerance = 1e-6
        rrf = residual / tolerance if residual > tolerance else 1.0
        target_sil = 0
        if rrf >= 10:
            target_sil = 1 if rrf < 100 else 2 if rrf < 1000 else 3 if rrf < 10000 else 4
            recommendations.append(
                {
                    "node_id": consequence,
                    "node_name": name,
                    "frequency": total,
                    "severity": severity_level,
                    "rrf": rrf,
                    "target_sil": target_sil,
                }
            )
        consequence_results.append(
            {
                "node_id": consequence,
                "node_name": name,
                "paths": path_details,
                "aggregation_steps": aggregation_steps,
                "frequency": total,
                "risk": {
                    "probability_level": probability_level,
                    "probability_description": probability_desc,
                    "severity_level": severity_level,
                    "severity_description": severity_desc,
                    "level": risk_level,
                    "action": action,
                },
                "lopa": {"pfd_dcs": 0.1, "pfd_relief_valve": 0.01, "residual_frequency": residual, "tolerance": tolerance, "rrf": rrf, "target_sil": target_sil},
            }
        )
    return {
        "nodes": nodes,
        "edges": edges,
        "forward_paths": forward_paths,
        "backward_paths": backward_paths,
        "consequences": consequence_results,
        "sil_recommendations": recommendations,
        "sis_required_nodes": [item["node_id"] for item in recommendations],
    }
