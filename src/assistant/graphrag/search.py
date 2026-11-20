"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

import networkx as nx

from assistant.infra.config import vocabulary
from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import MAP_SCHEMA, Answer, KnowledgeBase


def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на вопрос о конкретных сущностях по фактам из графа.

    Опорные сущности ищутся без модели: коды через `find_codes`, названия через `find_names`.
    При двух и более опорных сущностях берутся факты на путях длиной до трёх связей между ними
    и цепочки `REPLACED_BY` от деталей на этих путях. При одной берётся окрестность на глубину
    до двух связей. Модель получает пронумерованный список фактов и отвечает только по нему.
    Если в вопросе есть код, которого нет в графе, или фактов нет, возвращается
    `Answer(..., refused=True)` без вызова модели.
    """
    # TODO: опорные сущности, отбор фактов, запрос к модели, источники
    # raise NotImplementedError("Реализуйте local_search")

    vocab = vocabulary()
    codes = find_codes(question)
    
    for c in codes:
        ctype = code_type(c)
        node_id = f"{ctype}:{c}"
        if node_id not in kb.graph.nodes:
            return Answer(text="в базе знаний не найдено", sources=[], refused=True)

    anchor_codes = [c for c in codes if f"{code_type(c)}:{c}" in kb.graph.nodes]
    anchor_names = find_names(question, vocab)
    anchor_nodes = set()
    for c in anchor_codes:
        anchor_nodes.add(f"{code_type(c)}:{c}")
    for etype, ename in anchor_names:
        node_id = f"{etype}:{ename}"
        if node_id in kb.graph.nodes:
            anchor_nodes.add(node_id)
            
    if not anchor_nodes:
        return Answer(text="в базе знаний не найдено", sources=[], refused=True)

    facts = []
    sources = set()
    undirected = kb.graph.to_undirected()
    
    if len(anchor_nodes) >= 2:
        anchors_list = list(anchor_nodes)
        path_edges = set()
        for i in range(len(anchors_list)):
            for j in range(i + 1, len(anchors_list)):
                u = anchors_list[i]
                v = anchors_list[j]
                try:
                    paths = list(nx.all_simple_paths(undirected, u, v, cutoff=3))
                    for path in paths:
                        for k in range(len(path) - 1):
                            path_edges.add((path[k], path[k+1]))
                            path_edges.add((path[k+1], path[k]))
                except nx.NetworkXNoPath:
                    pass
        
        parts_on_path = set()
        for u, v in path_edges:
            if kb.graph.nodes[u]["type"] == "Part":
                parts_on_path.add(u)
            if kb.graph.nodes[v]["type"] == "Part":
                parts_on_path.add(v)
                
        for p in parts_on_path:
            for neighbor in kb.graph.successors(p):
                if kb.graph.edges[p, neighbor]["type"] == "REPLACED_BY":
                    path_edges.add((p, neighbor))
            for neighbor in kb.graph.predecessors(p):
                if kb.graph.edges[neighbor, p]["type"] == "REPLACED_BY":
                    path_edges.add((neighbor, p))
                    
        for u, v in path_edges:
            if kb.graph.has_edge(u, v):
                data = kb.graph.edges[u, v]
                facts.append(f"{kb.graph.nodes[u]['name']} -{data['type']}-> {kb.graph.nodes[v]['name']}")
                for src in data["sources"]:
                    sources.add(src.split("#")[0])
    else:
        anchor = list(anchor_nodes)[0]
        edges_in_neighborhood = set()
        for edge in nx.bfs_edges(undirected, anchor, depth_limit=2):
            edges_in_neighborhood.add(edge)
            edges_in_neighborhood.add((edge[1], edge[0]))
            
        for neighbor in undirected.neighbors(anchor):
            edges_in_neighborhood.add((anchor, neighbor))
            edges_in_neighborhood.add((neighbor, anchor))
            
        for u, v in edges_in_neighborhood:
            if kb.graph.has_edge(u, v):
                data = kb.graph.edges[u, v]
                facts.append(f"{kb.graph.nodes[u]['name']} -{data['type']}-> {kb.graph.nodes[v]['name']}")
                for src in data["sources"]:
                    sources.add(src.split("#")[0])

    if not facts:
        return Answer(text="в базе знаний не найдено", sources=[], refused=True)

    facts_text = "\n".join([f"{i+1}. {f}" for i, f in enumerate(facts)])
    prompt = f"""Ответь на вопрос, используя только следующие факты из базы знаний. 
Если в фактах есть информация о замене детали (REPLACED_BY), укажи самую новую (последнюю) деталь в цепочке замен.
Если ответа нет в фактах, напиши "в базе знаний не найдено".

Факты:
{facts_text}

Вопрос: {question}
"""
    response = llm.chat([{"role": "user", "content": prompt}], max_tokens=500)
    refused = "не найдено" in response.content.lower()
    return Answer(text=response.content.strip(), sources=sorted(list(sources)), facts=facts, refused=refused)


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага.

    Шаг map: для каждой сводки модель даёт частичный ответ и оценку полезности от 0 до 10.
    Шаг reduce: частичные ответы с оценкой выше 0 объединяются в итоговый ответ.
    Если таких ответов нет, возвращается `Answer(..., refused=True)`.
    """
    # TODO: частичные ответы по сводкам, объединение
    # raise NotImplementedError("Реализуйте global_search")

    import json
    partial_answers = []
    for comm in kb.communities:
        prompt = f"""Вопрос: {question}
Сводка сообщества: {comm.summary}

Дай краткий ответ на вопрос на основе этой сводки и оцени его полезность от 0 до 10.
Верни строго JSON по схеме: {{"answer": "текст ответа", "score": число}}
"""
        try:
            response = llm.chat([{"role": "user", "content": prompt}], json_schema=MAP_SCHEMA, max_tokens=300)
            data = json.loads(response.content)
            if data.get("score", 0) > 0:
                partial_answers.append(data.get("answer", ""))
        except Exception:
            continue
            
    if not partial_answers:
        return Answer(text="в базе знаний не найдено", sources=[], refused=True)
        
    combined = "\n".join([f"- {ans}" for ans in partial_answers])
    prompt = f"""Вопрос: {question}
Частичные ответы из разных сообществ:
{combined}

Сформируй итоговый связный ответ на вопрос, объединив полезную информацию из частичных ответов.
"""
    response = llm.chat([{"role": "user", "content": prompt}], max_tokens=500)
    return Answer(text=response.content.strip(), sources=[], refused=False)
