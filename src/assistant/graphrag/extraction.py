# src/assistant/graphrag/extraction.py
import json

from assistant.infra.config import vocabulary
from assistant.infra.grounding import canonical_name, find_codes
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import (
    ENTITY_TYPES,
    EXTRACTION_SCHEMA,
    RELATION_SIGNATURES,
    Chunk,
    Entity,
    ExtractionResult,
    Relation,
)

CODE_TYPES = ("Model", "Part", "ErrorCode", "Procedure")
RELATION_KEYS = ("subject_type", "subject", "type", "object_type", "object")


def _is_valid_payload(data) -> bool:
    """Проверка структуры ответа модели по EXTRACTION_SCHEMA без внешних библиотек."""
    if not isinstance(data, dict):
        return False
    entities = data.get("entities")
    relations = data.get("relations")
    if not isinstance(entities, list) or not isinstance(relations, list):
        return False
    for ent in entities:
        if not isinstance(ent, dict):
            return False
        if ent.get("type") not in ENTITY_TYPES or not isinstance(ent.get("name"), str):
            return False
    for rel in relations:
        if not isinstance(rel, dict):
            return False
        if any(not isinstance(rel.get(key), str) for key in RELATION_KEYS):
            return False
        if rel["type"] not in RELATION_SIGNATURES:
            return False
        if rel["subject_type"] not in ENTITY_TYPES or rel["object_type"] not in ENTITY_TYPES:
            return False
    return True


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
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
4. Ответ верни одним JSON-объектом с полями "entities" и "relations".

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
    data = None

    for attempt in range(2):
        raw = ""
        try:
            response = llm.chat(messages, json_schema=EXTRACTION_SCHEMA)
            raw = response.content or ""
            parsed = json.loads(raw)
            if not _is_valid_payload(parsed):
                raise ValueError("ответ не соответствует схеме извлечения")
            data = parsed
            break
        except Exception:
            if attempt == 1:
                return ExtractionResult(chunk.chunk_id, chunk.doc_id, dropped=0)
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": "Ответ не соответствует схеме JSON. Верни объект с полями "
                    '"entities" и "relations" строго по схеме.',
                }
            )

    if data is None:
        return ExtractionResult(chunk.chunk_id, chunk.doc_id, dropped=0)

    entities = []
    dropped = 0
    chunk_codes = set(find_codes(chunk.text))

    for ent in data["entities"]:
        etype = ent["type"]
        canon = canonical_name(etype, ent["name"], vocab)

        if canon is None:
            dropped += 1
            continue

        if etype in CODE_TYPES and canon not in chunk_codes:
            dropped += 1
            continue

        entities.append(Entity(etype, canon))

    relations = []
    for rel in data["relations"]:
        stype, otype = rel["subject_type"], rel["object_type"]
        scanon = canonical_name(stype, rel["subject"], vocab)
        ocanon = canonical_name(otype, rel["object"], vocab)

        s_valid = scanon is not None and (stype not in CODE_TYPES or scanon in chunk_codes)
        o_valid = ocanon is not None and (otype not in CODE_TYPES or ocanon in chunk_codes)

        if not (s_valid and o_valid):
            dropped += 1
            continue

        relation = Relation(Entity(stype, scanon), rel["type"], Entity(otype, ocanon))
        if relation.matches_schema():
            relations.append(relation)
        else:
            dropped += 1

    return ExtractionResult(chunk.chunk_id, chunk.doc_id, entities, relations, dropped)