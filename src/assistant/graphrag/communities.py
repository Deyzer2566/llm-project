"""Сообщества графа и их сводки. Требование FR-3."""

import networkx as nx

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Community


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества алгоритмом Лувена с фиксированным `seed`.

    Каждый узел входит ровно в одно сообщество. Поле `summary` здесь остаётся пустым.
    """
    # TODO: снять направление рёбер, разбить с учётом weight, вернуть Community с номерами по порядку
    # raise NotImplementedError("Реализуйте detect_communities")

    undirected = graph.to_undirected()
    communities_sets = nx.community.louvain_communities(undirected, weight="weight", seed=42)
    communities = []
    for idx, node_set in enumerate(communities_sets):
        communities.append(Community(community_id=idx, nodes=sorted(list(node_set))))
    return communities


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Сводка сообщества не длиннее 150 слов: какие сущности входят, как связаны, что встречается чаще.

    Все числа считает код по весам рёбер и передаёт модели готовыми. Модель их только формулирует.
    """
    # TODO: собрать факты и подсчёты сообщества, попросить модель написать сводку
    # raise NotImplementedError("Реализуйте summarize_community")
    subgraph = graph.subgraph(community.nodes)
    
    # Собираем полную статистику по сообществу
    nodes_by_type = {}
    for node in community.nodes:
        node_type = graph.nodes[node].get("type", "Unknown")
        node_name = graph.nodes[node].get("name", node)
        if node_type not in nodes_by_type:
            nodes_by_type[node_type] = []
        nodes_by_type[node_type].append(node_name)
    
    # Считаем веса для каждого типа связи
    relation_stats = {}
    component_weights = {}
    supplier_weights = {}
    model_failures = {}
    
    for u, v, data in subgraph.edges(data=True):
        rel_type = data.get("type", "UNKNOWN")
        weight = data.get("weight", 1)
        
        if rel_type not in relation_stats:
            relation_stats[rel_type] = []
        
        u_name = graph.nodes[u].get("name", u)
        v_name = graph.nodes[v].get("name", v)
        u_type = graph.nodes[u].get("type", "")
        v_type = graph.nodes[v].get("type", "")
        
        relation_stats[rel_type].append({
            "from": u_name,
            "to": v_name,
            "weight": weight
        })
        
        # Считаем веса для компонентов
        if u_type == "Component":
            component_weights[u_name] = component_weights.get(u_name, 0) + weight
        if v_type == "Component":
            component_weights[v_name] = component_weights.get(v_name, 0) + weight
        
        # Считаем веса для поставщиков
        if u_type == "Supplier":
            supplier_weights[u_name] = supplier_weights.get(u_name, 0) + weight
        if v_type == "Supplier":
            supplier_weights[v_name] = supplier_weights.get(v_name, 0) + weight
        
        # Считаем отказы моделей
        if rel_type == "FAILED":
            model_failures[v_name] = model_failures.get(v_name, 0) + weight
    
    # Формируем текст со всеми данными
    stats_lines = []
    stats_lines.append(f"В сообществе {len(community.nodes)} узлов и {subgraph.number_of_edges()} связей.")
    
    # Узлы по типам
    for node_type, names in sorted(nodes_by_type.items()):
        stats_lines.append(f"Узлы типа {node_type}: {', '.join(sorted(names)[:10])}")
    
    # Топ компонентов по весу
    if component_weights:
        top_components = sorted(component_weights.items(), key=lambda x: x[1], reverse=True)[:5]
        stats_lines.append("Топ компонентов по суммарному весу связей:")
        for name, weight in top_components:
            stats_lines.append(f"  - {name}: {weight}")
    
    # Топ поставщиков по весу
    if supplier_weights:
        top_suppliers = sorted(supplier_weights.items(), key=lambda x: x[1], reverse=True)[:5]
        stats_lines.append("Топ поставщиков по суммарному весу связей:")
        for name, weight in top_suppliers:
            stats_lines.append(f"  - {name}: {weight}")
    
    # Отказы моделей
    if model_failures:
        stats_lines.append("Отказы узлов в моделях:")
        for name, weight in sorted(model_failures.items(), key=lambda x: x[1], reverse=True)[:5]:
            stats_lines.append(f"  - {name}: {weight}")
    
    # Примеры связей каждого типа
    for rel_type, examples in sorted(relation_stats.items()):
        top_examples = sorted(examples, key=lambda x: x["weight"], reverse=True)[:3]
        stats_lines.append(f"Связи типа {rel_type} (топ по весу):")
        for ex in top_examples:
            stats_lines.append(f"  - {ex['from']} -> {ex['to']} (вес {ex['weight']})")
    
    stats_text = "\n".join(stats_lines)
    
    prompt = f"""Напиши краткую сводку сообщества графа знаний (не более 150 слов).
Используй только следующие данные:

{stats_text}

Опиши, какие сущности входят в это сообщество, как они связаны и особенности сообщества""" \
""": что встречается чаще всего, что отказывает чаще всего. Никаких сокращений типа "и другие", "и др."
"""
    response = llm.chat([{"role": "user", "content": prompt}], max_tokens=2048)
    summary = response.content.strip()
    words = summary.split()
    if len(words) > 150:
        summary = " ".join(words[:150])
    return summary
