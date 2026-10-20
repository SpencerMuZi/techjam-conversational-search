"""One response protocol for ranked candidates and one structured question."""
from __future__ import annotations

from .clarification import QUESTION_TEXT
from .slots import ATTRIBUTE_WORDS


CHINESE_QUESTIONS = {
    "category": "你想找哪一类商品？", "material": "你对材质有什么要求？",
    "color": "你有偏好的颜色吗？", "size": "你需要什么尺码或宽度？",
    "style": "你喜欢什么款式或版型？", "brand": "你有偏好的品牌吗？",
    "budget": "你的预算上限是多少？", "feature": "你最看重哪项功能？",
    "use_case": "你主要会在什么场景下使用？", "other": "还有哪些要求或偏好需要优先考虑？",
}


def question_for(attribute: str, user_message: str) -> str:
    if any("\u4e00" <= character <= "\u9fff" for character in user_message):
        return CHINESE_QUESTIONS[attribute]
    return QUESTION_TEXT.get(attribute, {
        "category": "What kind of product are you looking for?",
        "brand": "Do you have a preferred brand?",
    }[attribute] if attribute in {"category", "brand"} else "What else matters to you?")


def compose_response(state, message: str, recommendations: list[dict],
                     attribute: str | None = None, question: str | None = None) -> dict:
    # Questions have their own field in the tool contract and are appended here,
    # so the evaluator and the customer see the same question.
    if "?" in message or "？" in message:
        raise ValueError("Put the follow-up in question with ask_attribute, not in message")
    if attribute is None:
        if question is not None:
            raise ValueError("question requires ask_attribute")
    else:
        if attribute not in ATTRIBUTE_WORDS:
            raise ValueError("Invalid ask_attribute")
        if attribute in state.no_preference_attributes:
            raise ValueError("User already declined a preference for this attribute")
        if not question or question.count("?") + question.count("？") > 1:
            raise ValueError("Provide exactly one follow-up question")
        latest_user = next((text for role, text in reversed(state.history) if role == "user"), "")
        # A model can label a sleeve-length question as "other". Rendering from
        # the attribute prevents that semantic mismatch with the simulator/UI.
        question = question_for(attribute, latest_user)
        message = (message.rstrip() + "\n\n" + question.strip()).strip()
    return {"message": message, "ask_attribute": attribute, "recommendations": recommendations}
