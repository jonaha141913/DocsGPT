from application.llm.handlers.base import LLMHandler
from application.llm.handlers.openai import OpenAILLMHandler
from application.llm.handlers.google import GoogleLLMHandler


class LLMHandlerCreator:
    @classmethod
    def _get_handlers(cls) -> dict:
        return {
            "openai": OpenAILLMHandler,
            "google": GoogleLLMHandler,
            "default": OpenAILLMHandler,
        }

    @classmethod
    def create_handler(cls, llm_type: str, *args, **kwargs) -> LLMHandler:
        handlers = cls._get_handlers()
        handler_class = handlers.get(llm_type.lower())
        if not handler_class:
            handler_class = OpenAILLMHandler
        return handler_class(*args, **kwargs)
