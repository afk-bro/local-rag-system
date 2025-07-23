"""
Prompt templates for the RAG system.
"""

from langchain.prompts import PromptTemplate

# RAG prompt template as specified by the user
RAG_PROMPT_TEMPLATE = """You are an expert assistant helping answer questions based on provided context.

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
Give a concise and factual answer using only the information from the context.
If the answer cannot be found in the context, respond with: "The answer is not in the provided documents."
"""

# Create LangChain PromptTemplate
rag_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=RAG_PROMPT_TEMPLATE
)

# Additional prompt templates for different use cases
SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on provided context. Always be accurate and cite your sources when possible."""

SUMMARIZATION_PROMPT = """Please provide a concise summary of the following text:

{text}

Summary:"""

summarization_prompt = PromptTemplate(
    input_variables=["text"],
    template=SUMMARIZATION_PROMPT
)
