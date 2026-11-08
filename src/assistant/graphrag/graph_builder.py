"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

import networkx as nx

from assistant.infra.schemas import ExtractionResult


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты по всем чанкам в один направленный граф.

    Узел: идентификатор `Entity.node_id`, атрибуты `type`, `name`, `sources` (список chunk_id).
    Ребро: атрибуты `type`, `weight` (число разных чанков, где связь найдена), `sources`.
    Связи, не соответствующие схеме (`Relation.matches_schema`), отбрасываются. Если между двумя
    деталями есть `REPLACED_BY` в обе стороны, остаётся связь с большим весом, при равенстве
    отбрасываются обе. Число таких конфликтов записывается в `graph.graph["conflicts"]`.
    """
    # TODO: узлы, рёбра с весами, проверка схемы, разбор встречных замен
    # raise NotImplementedError("Реализуйте build_graph")

    graph = nx.DiGraph(conflicts=0)
    
    for res in results:
        for ent in res.entities:
            node_id = ent.node_id
            if node_id not in graph:
                graph.add_node(node_id, type=ent.type, name=ent.name, sources=[])
            if res.chunk_id not in graph.nodes[node_id]["sources"]:
                graph.nodes[node_id]["sources"].append(res.chunk_id)
                
        for rel in res.relations:
            if not rel.matches_schema():
                continue
            u = rel.subject.node_id
            v = rel.object.node_id
            
            if graph.has_edge(u, v):
                if res.chunk_id not in graph.edges[u, v]["sources"]:
                    graph.edges[u, v]["sources"].append(res.chunk_id)
                    graph.edges[u, v]["weight"] += 1
            else:
                graph.add_edge(u, v, type=rel.type, weight=1, sources=[res.chunk_id])

    conflicts = 0
    edges_to_remove = []
    processed = set()
    
    for u, v, data in list(graph.edges(data=True)):
        if data["type"] == "REPLACED_BY" and graph.has_edge(v, u) and graph.edges[v, u]["type"] == "REPLACED_BY":
            pair = tuple(sorted([u, v]))
            if pair in processed:
                continue
            processed.add(pair)
            
            w1 = data["weight"]
            w2 = graph.edges[v, u]["weight"]
            if w1 > w2:
                edges_to_remove.append((v, u))
            elif w2 > w1:
                edges_to_remove.append((u, v))
            else:
                edges_to_remove.append((u, v))
                edges_to_remove.append((v, u))
            conflicts += 1

    for u, v in edges_to_remove:
        if graph.has_edge(u, v):
            graph.remove_edge(u, v)
            
    graph.graph["conflicts"] = conflicts
    return graph
