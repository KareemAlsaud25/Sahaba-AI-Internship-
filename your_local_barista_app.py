import streamlit as st
from LLM_Model import llm
from RAG import retriever
from prompts import prompt

st.set_page_config(page_title="Your Local Barista", page_icon="☕")
st.title("☕ Your Local Barista")
st.caption(
    "Ask me anything about espresso, milk steaming, coffee beans, or "
    "barista tools. I only answer from the coffee knowledge base — "
    "I won't make things up outside of it."
)

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
user_question = st.chat_input("Ask a coffee question...")

if user_question:
    # Show and store the user's message
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Build a plain-text chat history string from prior turns
    # (excluding the message we just added)
    history_text = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}"
        for msg in st.session_state.messages[:-1]
    )
    if not history_text:
        history_text = "(no previous messages)"

    # Retrieve relevant chunks for this question
    retrieved_docs = retriever.invoke(user_question)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # Fill in the prompt template with history + context + question
    filled_prompt = prompt.format(
        chat_history=history_text,
        context=context_text,
        question=user_question,
    )

    # Get the LLM's answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = llm.invoke(filled_prompt)
            answer = response.content
            st.markdown(answer)

    # Store the assistant's reply in history
    st.session_state.messages.append({"role": "assistant", "content": answer})