"""Извлечение сущностей и связей из чанка. Требование FR-1."""

import json
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Chunk, ExtractionResult, Entity, Relation, EXTRACTION_SCHEMA
from assistant.infra.grounding import find_codes, canonical_name
from assistant.infra.config import vocabulary

def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает сущности и связи чанка строго по схеме графа.

    Модель вызывается через `llm.chat(..., json_schema=EXTRACTION_SCHEMA)`. Запрос описывает схему
    с направлением каждой связи и содержит хотя бы один пример. Всё, что не подтверждается текстом,
    отбрасывается и считается в `dropped`: код, которого нет в чанке, название не из словаря,
    связь с отброшенным концом. Пригодятся `find_codes` и `canonical_name` из `assistant.infra.grounding`.
    Функция не падает ни на каком ответе модели: одна повторная попытка, затем пустой результат.
    """
    # TODO: запрос к модели, разбор JSON, проверки по тексту чанка и словарю
    # raise NotImplementedError("Реализуйте extract_chunk")

    vocab = vocabulary()
    prompt = f"""Ты извлекаешь сущности и связи из текста чанка строго по заданной схеме.
Схема связей (направление важно!):
- Model BELONGS_TO ProductLine
- Model HAS_PART Part
- Part IS_A Component
- Part REPLACED_BY Part (старая деталь заменяется на новую)
- Part SUPPLIED_BY Supplier
- ErrorCode INDICATES Component
- ErrorCode FIXED_BY Procedure
- Model FAILED Component

Правила:
1. Извлекай только то, что явно есть в тексте.
2. Коды (Model, Part, ErrorCode, Procedure) должны точно совпадать с текстом (например, P-1042, E-17).
3. Названия (ProductLine, Component, Supplier) должны быть из словаря: {vocab}.

Пример:
Текст: "Деталь P-1042 (пусковое реле) снята с производства. Вместо неё устанавливается деталь P-1042B. Поставщик – «ХолодКомплект»."
Результат:
{{
  "entities": [
    {{"type": "Part", "name": "P-1042"}},
    {{"type": "Part", "name": "P-1042B"}},
    {{"type": "Component", "name": "пусковое реле"}},
    {{"type": "Supplier", "name": "ХолодКомплект"}}
  ],
  "relations": [
    {{"subject_type": "Part", "subject": "P-1042", "type": "REPLACED_BY", "object_type": "Part", "object": "P-1042B"}},
    {{"subject_type": "Part", "subject": "P-1042", "type": "IS_A", "object_type": "Component", "object": "пусковое реле"}},
    {{"subject_type": "Part", "subject": "P-1042B", "type": "SUPPLIED_BY", "object_type": "Supplier", "object": "ХолодКомплект"}}
  ]
}}

Текст чанка:
{chunk.text}
"""
    messages = [{"role": "user", "content": prompt}]
    
    for attempt in range(2):
        try:
            response = llm.chat(messages, json_schema=EXTRACTION_SCHEMA)
            data = json.loads(response.content)
            if not isinstance(data, dict):
                raise ValueError("JSON is not a dictionary")
            break
        except Exception:
            if attempt == 1:
                return ExtractionResult(chunk.chunk_id, chunk.doc_id, dropped=0)
            messages.append({"role": "assistant", "content": response.content if 'response' in locals() else ""})
            messages.append({"role": "user", "content": "Ошибка формата JSON. Попробуй ещё раз, строго следуя схеме."})
    else:
        return ExtractionResult(chunk.chunk_id, chunk.doc_id, dropped=0)

    entities = []
    valid_nodes = set()
    dropped = 0
    chunk_codes = set(find_codes(chunk.text))
    
    for ent in data.get("entities", []):
        etype = ent.get("type")
        ename = ent.get("name", "")
        canon = canonical_name(etype, ename, vocab)
        
        if canon is None:
            dropped += 1
            continue
            
        if etype in ("Model", "Part", "ErrorCode", "Procedure"):
            if canon not in chunk_codes:
                dropped += 1
                continue
                
        entities.append(Entity(etype, canon))
        valid_nodes.add(f"{etype}:{canon}")

    relations = []
    for rel in data.get("relations", []):
        stype = rel.get("subject_type")
        sname = rel.get("subject", "")
        rtype = rel.get("type")
        otype = rel.get("object_type")
        oname = rel.get("object", "")
        
        scanon = canonical_name(stype, sname, vocab)
        ocanon = canonical_name(otype, oname, vocab)
        
        s_valid = scanon is not None
        o_valid = ocanon is not None
        
        if stype in ("Model", "Part", "ErrorCode", "Procedure") and scanon:
            s_valid = scanon in chunk_codes
        if otype in ("Model", "Part", "ErrorCode", "Procedure") and ocanon:
            o_valid = ocanon in chunk_codes
            
        if s_valid and o_valid:
            subj = Entity(stype, scanon)
            obj = Entity(otype, ocanon)
            relation = Relation(subj, rtype, obj)
            if relation.matches_schema():
                relations.append(relation)
            else:
                dropped += 1
        else:
            dropped += 1

    return ExtractionResult(chunk.chunk_id, chunk.doc_id, entities, relations, dropped)