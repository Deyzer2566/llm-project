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
    
    component_counts = {}
    supplier_counts = {}
    max_weight = 0
    top_relation = ""
    
    subgraph = graph.subgraph(community.nodes)
    for u, v, data in subgraph.edges(data=True):
        w = data["weight"]
        if w > max_weight:
            max_weight = w
            top_relation = f"{graph.nodes[u]['name']} {data['type']} {graph.nodes[v]['name']}"
        
        u_type = graph.nodes[u]["type"]
        v_type = graph.nodes[v]["type"]
        if u_type == "Component":
            component_counts[graph.nodes[u]["name"]] = component_counts.get(graph.nodes[u]["name"], 0) + w
        if v_type == "Component":
            component_counts[graph.nodes[v]["name"]] = component_counts.get(graph.nodes[v]["name"], 0) + w
            
        if u_type == "Supplier":
            supplier_counts[graph.nodes[u]["name"]] = supplier_counts.get(graph.nodes[u]["name"], 0) + w
        if v_type == "Supplier":
            supplier_counts[graph.nodes[v]["name"]] = supplier_counts.get(graph.nodes[v]["name"], 0) + w

    top_component = max(component_counts, key=component_counts.get) if component_counts else "нет"
    top_supplier = max(supplier_counts, key=supplier_counts.get) if supplier_counts else "нет"
    
    prompt = f"""Напиши краткую сводку сообщества графа знаний (не более 150 слов).
В сообществе {len(community.nodes)} узлов.
Самая частая связь: {top_relation} (вес {max_weight}).
Самый упоминаемый узел типа Component: {top_component}.
Самый упоминаемый узел типа Supplier: {top_supplier}.

Опиши, какие сущности входят в это сообщество, как они связаны и что встречается чаще всего, используя предоставленные числа.
"""
    response = llm.chat([{"role": "user", "content": prompt}], max_tokens=300)
    return response.content.strip()
