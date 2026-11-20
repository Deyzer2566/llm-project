"""Инструменты агента. Требование FR-6."""

import re
from collections.abc import Callable

from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected, ErpUnavailable
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]

# Описания пяти инструментов в формате function calling:
# search_knowledge, summarize_topic, check_stock, get_price, create_service_order.
# TODO: заполните список. У параметра warehouse четыре склада задаются перечислением.
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "Поиск фактов в графе знаний по конкретному вопросу.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_topic",
            "description": "Поиск обзорной информации по всему корпусу документов.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": "Проверить остаток детали на складе.",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {"type": "string", "description": "Номер детали, например P-1042B"},
                    "warehouse": {"type": "string", "enum": WAREHOUSES},
                },
                "required": ["part_id", "warehouse"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_price",
            "description": "Узнать цену детали.",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {"type": "string", "description": "Номер детали, например P-1042B"},
                },
                "required": ["part_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_service_order",
            "description": "Создать заказ на деталь (требует подтверждения).",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {"type": "string", "description": "Номер детали"},
                    "warehouse": {"type": "string", "enum": WAREHOUSES},
                    "qty": {"type": "integer", "minimum": 1},
                    "model": {"type": "string", "description": "Модель оборудования"},
                },
                "required": ["part_id", "warehouse", "qty", "model"],
            },
        },
    },
]

def _validate_part_id(part_id: str) -> bool:
    return bool(re.match(r"^P-\d+[A-Z]?$", str(part_id)))

def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Словарь «имя инструмента -> функция». Каждая функция возвращает словарь.

    Функция проверяет свои аргументы до обращения к ERP: склад из списка, номер детали вида P-1042
    или P-1042B, количество больше нуля. При ошибке возвращает {"error": "..."} и ничего не выполняет.
    `ErpNotFound` и `ErpRejected` превращаются в {"error": "..."} с разным текстом, чтобы исходы
    оставались различимыми. `ErpUnavailable` передаётся дальше: повторы делает агент (FR-8).
    """
    # TODO: пять функций поверх local_search, global_search и клиента ERP
    # raise NotImplementedError("Реализуйте build_tools")
    def search_knowledge(question: str) -> dict:
        ans = local_search(question, kb, llm)
        return {"answer": ans.text, "facts": ans.facts, "sources": ans.sources, "refused": ans.refused}

    def summarize_topic(question: str) -> dict:
        ans = global_search(question, kb, llm)
        return {"answer": ans.text, "sources": ans.sources, "refused": ans.refused}

    def check_stock(part_id: str, warehouse: str) -> dict:
        if not _validate_part_id(part_id):
            return {"error": f"Неверный формат part_id: {part_id}"}
        if warehouse not in WAREHOUSES:
            return {"error": f"Неизвестный склад: {warehouse}. Допустимые: {', '.join(WAREHOUSES)}"}
        try:
            return erp.get_stock(part_id, warehouse)
        except ErpNotFound as e:
            return {"error": f"Деталь не найдена: {e}"}
        except ErpRejected as e:
            return {"error": f"Запрос отклонён: {e}"}
        except ErpUnavailable as e:
            raise e

    def get_price(part_id: str) -> dict:
        if not _validate_part_id(part_id):
            return {"error": f"Неверный формат part_id: {part_id}"}
        try:
            return erp.get_price(part_id)
        except ErpNotFound as e:
            return {"error": f"Деталь не найдена: {e}"}
        except ErpRejected as e:
            return {"error": f"Запрос отклонён: {e}"}
        except ErpUnavailable as e:
            raise e

    def create_service_order(part_id: str, warehouse: str, qty: int, model: str) -> dict:
        if not _validate_part_id(part_id):
            return {"error": f"Неверный формат part_id: {part_id}"}
        if warehouse not in WAREHOUSES:
            return {"error": f"Неизвестный склад: {warehouse}"}
        if not isinstance(qty, int) or qty <= 0:
            return {"error": f"Количество должно быть целым числом больше 0, получено: {qty}"}
        try:
            return erp.create_order(part_id, warehouse, qty, model)
        except ErpNotFound as e:
            return {"error": f"Деталь не найдена: {e}"}
        except ErpRejected as e:
            return {"error": f"Заказ отклонён: {e}"}
        except ErpUnavailable as e:
            raise e

    return {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": check_stock,
        "get_price": get_price,
        "create_service_order": create_service_order,
    }