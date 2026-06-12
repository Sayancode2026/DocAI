import os
import sys
import json
from dotenv import load_dotenv
from utils.config_loader import load_config
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException


# GOOGLE_API_KEY needed only when LLM_PROVIDER=google
# Embeddings are now HuggingFace (local, free, no API key)
PROVIDER_REQUIRED_KEYS = {
    "google": ["GOOGLE_API_KEY"],
    "groq":   ["GROQ_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
}


class ApiKeyManager:
    ALL_KEYS = ["GROQ_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"]

    def __init__(self):
        self.api_keys = {}
        raw = os.getenv("API_KEYS")

        # ECS / Secrets Manager: single JSON blob
        if raw:
            try:
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("API_KEYS is not a valid JSON object")
                self.api_keys = parsed
                log.info("Loaded API_KEYS from ECS secret")
            except Exception as e:
                log.warning("Failed to parse API_KEYS as JSON", error=str(e))

        # Fallback: individual env vars (local .env)
        for key in self.ALL_KEYS:
            if not self.api_keys.get(key):
                env_val = os.getenv(key)
                if env_val:
                    self.api_keys[key] = env_val
                    log.info(f"Loaded {key} from individual env var")

        # Validate only keys needed for active provider
        active_provider = os.getenv("LLM_PROVIDER", "groq")
        required = PROVIDER_REQUIRED_KEYS.get(active_provider, [])
        missing = [k for k in required if not self.api_keys.get(k)]

        if missing:
            log.error("Missing required API keys", missing_keys=missing, provider=active_provider)
            raise DocumentPortalException("Missing API keys", sys)

        loaded = {k: v[:6] + "..." for k, v in self.api_keys.items() if v}
        log.info("API keys loaded", keys=loaded, active_provider=active_provider)

    def get(self, key: str) -> str:
        val = self.api_keys.get(key)
        if not val:
            raise KeyError(f"API key for '{key}' is missing. Check your .env file.")
        return val


class ModelLoader:
    """
    Loads embedding models and LLMs based on config and environment.

    Embeddings: HuggingFace (local, free, no API key required)
    LLM: switches based on LLM_PROVIDER env var: groq | google | openai
    """

    def __init__(self):
        if os.getenv("ENV", "local").lower() != "production":
            load_dotenv()
            log.info("Running in LOCAL mode: .env loaded")
        else:
            log.info("Running in PRODUCTION mode")

        self.api_key_mgr = ApiKeyManager()
        self.config = load_config()
        log.info("YAML config loaded", config_keys=list(self.config.keys()))

    def load_embeddings(self):
        """
        Load HuggingFace embeddings — runs locally, no API key needed.
        Model downloads once and is cached at ~/.cache/huggingface/
        """
        try:
            model_name = self.config["embedding_model"]["model_name"]
            log.info("Loading HuggingFace embedding model", model=model_name)
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as e:
            log.error("Error loading embedding model", error=str(e))
            raise DocumentPortalException("Failed to load embedding model", sys)

    def load_llm(self):
        """
        Load LLM based on LLM_PROVIDER env var.
        groq   → ChatGroq   (free, fast)
        google → ChatGoogleGenerativeAI (Gemini)
        openai → ChatOpenAI
        """
        try:
            llm_block = self.config["llm"]
            provider_key = os.getenv("LLM_PROVIDER", "groq")

            if provider_key not in llm_block:
                log.error("LLM provider not found in config", provider=provider_key)
                raise ValueError(
                    f"LLM provider '{provider_key}' not found in config.yaml. "
                    f"Available: {list(llm_block.keys())}"
                )

            llm_config  = llm_block[provider_key]
            provider    = llm_config.get("provider")
            model_name  = llm_config.get("model_name")
            temperature = llm_config.get("temperature", 0.2)
            max_tokens  = llm_config.get("max_output_tokens", 2048)

            log.info("Loading LLM", provider=provider, model=model_name)

            if provider == "google":
                return ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=self.api_key_mgr.get("GOOGLE_API_KEY"),
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )

            elif provider == "groq":
                return ChatGroq(
                    model=model_name,
                    api_key=self.api_key_mgr.get("GROQ_API_KEY"),  # type: ignore
                    temperature=temperature,
                )

            elif provider == "openai":
                return ChatOpenAI(
                    model=model_name,
                    api_key=self.api_key_mgr.get("OPENAI_API_KEY"),
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            else:
                log.error("Unsupported LLM provider", provider=provider)
                raise ValueError(f"Unsupported LLM provider: {provider}")

        except Exception as e:
            log.error("Error loading LLM", error=str(e))
            raise DocumentPortalException("Failed to load LLM", sys)


if __name__ == "__main__":
    loader = ModelLoader()

    embeddings = loader.load_embeddings()
    print(f"Embedding Model Loaded: {embeddings}")
    result = embeddings.embed_query("Hello, how are you?")
    print(f"Embedding Result (first 5 dims): {result[:5]}")

    llm = loader.load_llm()
    print(f"LLM Loaded: {llm}")
    result = llm.invoke("Hello, how are you?")
    print(f"LLM Result: {result.content}")
