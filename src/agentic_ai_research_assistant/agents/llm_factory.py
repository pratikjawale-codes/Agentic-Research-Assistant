import re
from typing import Optional, Any
from pydantic import BaseModel
from agentic_ai_research_assistant.config import settings

class LLMFactory:
    """
    Factory for LLM providers.
    Supports OpenAI, Google Gemini, and Anthropic.
    Provides intelligent local execution when API keys are not yet configured by the user.
    """

    @staticmethod
    def get_chat_model(temperature: float = 0.0, role: str = "general"):
        if settings.OPENAI_API_KEY:
            from langchain_openai import ChatOpenAI
            model_name = "gpt-4o" if role in ["supervisor", "report"] else "gpt-4o-mini"
            return ChatOpenAI(model=model_name, temperature=temperature, api_key=settings.OPENAI_API_KEY)

        if settings.GEMINI_API_KEY:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=temperature, google_api_key=settings.GEMINI_API_KEY)

        if settings.ANTHROPIC_API_KEY:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=temperature, api_key=settings.ANTHROPIC_API_KEY)

        # Returns None to indicate that local deterministic orchestrator handles reasoning
        return None
