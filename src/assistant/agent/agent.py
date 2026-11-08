"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

import re
import json
from collections.abc import Callable

from assistant.infra.llm import LLMClient, tool_message
from assistant.infra.schemas import AgentResult, ToolStep
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.grounding import find_codes
from assistant.infra.evaluation import fabricated_facts
from assistant.agent.tools import TOOL_SCHEMAS


def run_agent(
    question: str,
    llm: LLMClient,
    tools: dict[str, Callable[..., dict]],
    *,
    max_steps: int = 6,
    confirm: bool = False,
) -> AgentResult:
    """Ведёт диалог с моделью в цикле «рассуждение, вызов инструмента, наблюдение».

    Модель получает `TOOL_SCHEMAS` через `llm.chat(..., tools=...)`. Каждый вызов записывается
    в `steps` как `ToolStep`. Агент сам проверяет номер детали в аргументах (он должен встречаться
    в вопросе или в результатах прошлых инструментов), количество в заказе (должно быть названо
    пользователем) и повторные вызовы. При `ErpUnavailable` вызов повторяется до двух раз.
    Заказ создаётся только при `confirm=True`, иначе агент возвращает `needs_confirmation=True`
    и `pending_action`. Перед выдачей ответа коды, цены и остатки сверяются с вопросом
    и результатами инструментов.
    """
    # TODO: цикл, проверки аргументов, повторы, подтверждение, проверка итогового ответа
    # raise NotImplementedError("Реализуйте run_agent")
    messages = [{"role": "user", "content": question}]
    steps = []
    sources = set()
    
    known_parts = set(find_codes(question))
    tool_results_text = ""
    
    qty_match = re.search(r"(\d+)\s*(?:штук|штуки|шт)", question, re.IGNORECASE)
    qty_in_question = qty_match.group(1) if qty_match else None

    for step_num in range(max_steps):
        response = llm.chat(messages, tools=TOOL_SCHEMAS)
        
        if not response.tool_calls:
            break
            
        for call in response.tool_calls:
            tool_name = call.name
            args = call.arguments
            
            step = ToolStep(tool=tool_name, args=args)
            
            if "_raw" in args:
                step.error = "Нечитаемые аргументы инструмента."
                steps.append(step)
                messages.append(response.as_message())
                messages.append(tool_message(call, {"error": step.error}))
                continue
            
            if "part_id" in args:
                if str(args["part_id"]) not in known_parts:
                    step.error = f"Неизвестная деталь {args['part_id']}. Сначала используйте search_knowledge, чтобы найти нужную деталь."
                    steps.append(step)
                    messages.append(response.as_message())
                    messages.append(tool_message(call, {"error": step.error}))
                    continue
            
            if tool_name == "create_service_order":
                if "qty" in args and qty_in_question is None:
                    step.error = "Количество не указано в вопросе. Уточните у пользователя, сколько штук заказать."
                    steps.append(step)
                    messages.append(response.as_message())
                    messages.append(tool_message(call, {"error": step.error}))
                    continue

            if tool_name == "create_service_order" and not confirm:
                steps.append(step)
                return AgentResult(
                    answer="Заказ готов к оформлению. Требуется подтверждение.",
                    steps=steps,
                    sources=sorted(list(sources)),
                    refused=False,
                    needs_confirmation=True,
                    pending_action={"tool": "create_service_order", "args": args},
                )
            
            if tool_name == "create_service_order":
                if "qty" in args and qty_in_question is None:
                    step.error = "Количество не указано в вопросе. Уточните у пользователя, сколько штук заказать."
                    steps.append(step)
                    messages.append(response.as_message())
                    messages.append(tool_message(call, {"error": step.error}))
                    continue

            cached_result = None
            for prev_step in steps:
                if prev_step.tool == tool_name and prev_step.args == args and prev_step.error is None:
                    cached_result = prev_step.result
                    break
            
            if cached_result is not None:
                step.result = cached_result
                steps.append(step)
                messages.append(response.as_message())
                messages.append(tool_message(call, cached_result))
                continue

            attempts = 0
            max_retries = 3
            result = None
            while attempts < max_retries:
                try:
                    if tool_name not in tools:
                        step.error = f"Неизвестный инструмент: {tool_name}"
                        break
                    
                    result = tools[tool_name](**args)
                    if isinstance(result, dict) and "error" in result:
                        step.error = result["error"]
                    else:
                        step.result = result
                    break
                except ErpUnavailable:
                    attempts += 1
                    if attempts == max_retries:
                        step.error = "Данные ERP получить не удалось после нескольких попыток."
            
            steps.append(step)
            messages.append(response.as_message())
            
            if step.result is not None:
                messages.append(tool_message(call, step.result))
                result_text = json.dumps(step.result, ensure_ascii=False)
                tool_results_text += result_text + " "
                for code in find_codes(result_text):
                    known_parts.add(code)
                if "sources" in step.result and isinstance(step.result["sources"], list):
                    for src in step.result["sources"]:
                        sources.add(src)
            else:
                messages.append(tool_message(call, {"error": step.error}))

    final_answer = response.content.strip() if response.content else "Работа прервана: исчерпан лимит шагов."
    
    def verify_answer(ans: str) -> tuple[bool, str]:
        known_codes = set(find_codes(question) + find_codes(tool_results_text))
        fabricated = fabricated_facts(ans, question, known_codes, tool_results_text)
        if fabricated:
            return False, f"Выдуманные факты: {fabricated}"
        return True, ""

    is_valid, error_msg = verify_answer(final_answer)
    if not is_valid and final_answer:
        correction_prompt = f"Твой ответ содержит непроверенные факты: {error_msg}. Исправь ответ, используя только данные из вопроса и результатов инструментов."
        messages.append({"role": "assistant", "content": final_answer})
        messages.append({"role": "user", "content": correction_prompt})
        correction_response = llm.chat(messages)
        final_answer = correction_response.content.strip()
        
        is_valid2, error_msg2 = verify_answer(final_answer)
        if not is_valid2:
            return AgentResult(
                answer="в базе знаний не найдено",
                steps=steps,
                sources=sorted(list(sources)),
                refused=True,
            )

    is_refused = any(step.result and isinstance(step.result, dict) and step.result.get("refused") for step in steps) or "не найдено" in final_answer.lower()

    return AgentResult(
        answer=final_answer,
        steps=steps,
        sources=sorted(list(sources)),
        refused=is_refused,
    )
