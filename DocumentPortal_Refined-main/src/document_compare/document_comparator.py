import sys
from dotenv import load_dotenv
import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from utils.model_loader import ModelLoader
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException
from prompt.prompt_library import PROMPT_REGISTRY
from src.model.models import SummaryResponse, PromptType

# Groq free tier: 12k TPM limit
# Two docs + prompt + output must fit. Give ~4000 chars per doc = ~2000 tokens each
MAX_CHARS_PER_DOC = 4000


class DocumentComparatorLLM:
    def __init__(self):
        load_dotenv()
        self.loader = ModelLoader()
        self.llm = self.loader.load_llm()
        self.parser = JsonOutputParser(pydantic_object=SummaryResponse)
        self.prompt = PROMPT_REGISTRY[PromptType.DOCUMENT_COMPARISON.value]
        self.chain = self.prompt | self.llm | self.parser
        log.info("DocumentComparatorLLM initialized", model=self.llm)

    def compare_documents(self, combined_docs: str) -> pd.DataFrame:
        try:
            # Split combined text back into two docs and truncate each
            combined_docs = self._truncate_combined(combined_docs)

            inputs = {
                "combined_docs": combined_docs,
                "format_instruction": self.parser.get_format_instructions()
            }
            log.info("Invoking document comparison LLM chain")
            response = self.chain.invoke(inputs)
            log.info("Chain invoked successfully", response_preview=str(response)[:200])
            return self._format_response(response)
        except Exception as e:
            log.error("Error in compare_documents", error=str(e))
            raise DocumentPortalException("Error comparing documents", sys)

    def _truncate_combined(self, combined_docs: str) -> str:
        """
        Split on 'Document:' marker and truncate each doc to MAX_CHARS_PER_DOC.
        Keeps comparison meaningful while staying within Groq token limits.
        """
        parts = combined_docs.split("Document:")
        truncated_parts = []
        for part in parts:
            if not part.strip():
                continue
            if len(part) > MAX_CHARS_PER_DOC:
                part = part[:MAX_CHARS_PER_DOC] + "\n...[truncated for token limit]"
                log.info("Document truncated for comparison", truncated_chars=MAX_CHARS_PER_DOC)
            truncated_parts.append("Document:" + part)

        result = "\n\n".join(truncated_parts)
        log.info("Combined docs prepared", total_chars=len(result))
        return result

    def _format_response(self, response_parsed) -> pd.DataFrame:
        try:
            df = pd.DataFrame(response_parsed)
            return df
        except Exception as e:
            log.error("Error formatting response into DataFrame", error=str(e))
            raise DocumentPortalException("Error formatting response", sys)