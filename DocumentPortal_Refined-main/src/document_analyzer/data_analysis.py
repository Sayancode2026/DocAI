import os
import sys
from utils.model_loader import ModelLoader
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException
from src.model.models import *
from langchain_core.output_parsers import JsonOutputParser
from prompt.prompt_library import PROMPT_REGISTRY  # type: ignore

MAX_CHARS_FOR_ANALYSIS = 8000  # ~2000 tokens — safe for Groq free tier


class DocumentAnalyzer:
    """
    Analyzes documents using a pre-trained model.
    """

    def __init__(self):
        try:
            self.loader = ModelLoader()
            self.llm = self.loader.load_llm()
            self.parser = JsonOutputParser(pydantic_object=Metadata)
            self.prompt = PROMPT_REGISTRY["document_analysis"]
            log.info("DocumentAnalyzer initialized successfully")
        except Exception as e:
            log.error(f"Error initializing DocumentAnalyzer: {e}")
            raise DocumentPortalException("Error in DocumentAnalyzer initialization", sys)

    def analyze_document(self, document_text: str) -> dict:
        try:
            # JsonOutputParser handles output directly — no OutputFixingParser needed
            chain = self.prompt | self.llm | self.parser

            log.info("Meta-data analysis chain initialized")

            original_len = len(document_text)
            if original_len > MAX_CHARS_FOR_ANALYSIS:
                document_text = document_text[:MAX_CHARS_FOR_ANALYSIS]
                log.info(
                    "Document text truncated for analysis",
                    original_chars=original_len,
                    truncated_chars=MAX_CHARS_FOR_ANALYSIS,
                )

            response = chain.invoke({
                "format_instructions": self.parser.get_format_instructions(),
                "document_text": document_text,
            })

            log.info("Metadata extraction successful", keys=list(response.keys()))
            return response

        except Exception as e:
            log.error("Metadata analysis failed", error=str(e))
            raise DocumentPortalException("Metadata extraction failed", sys)