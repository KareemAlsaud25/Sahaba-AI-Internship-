from langchain_core.prompts import PromptTemplate

RAG_PROMPT_TEMPLATE = """You are a helpful assistant answering questions about coffee and barista knowledge.
Use ONLY the context below to answer the question. Do not use any outside knowledge.

Use the chat history to understand follow-up questions that refer back to
earlier parts of the conversation, but still answer strictly using the
context provided below.

If the answer cannot be found in the context, respond exactly with:
"I don't know based on the provided document."

Chat History:
{chat_history}

Context:
{context}

Question:
{question}

Answer:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template=RAG_PROMPT_TEMPLATE,
)