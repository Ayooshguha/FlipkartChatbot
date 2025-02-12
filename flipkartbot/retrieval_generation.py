from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from flipkartbot.data_ingestion import data_ingestion
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv("D:\flipkartbot\FlipkartChatbot\.env")

GROQ_API_KEY = "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"  # Manually set the API key
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Check if required environment variables are set
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")
HF_TOKEN = os.getenv("HF_TOKEN")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the .env file or environment variables.")
if not all([ASTRA_DB_API_ENDPOINT, ASTRA_DB_APPLICATION_TOKEN, ASTRA_DB_KEYSPACE, HF_TOKEN]):
    raise ValueError("One or more required environment variables are missing. Check your .env file.")

# Set the environment variable
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Initialize the Groq model
model = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0.5)

# Chat history and store
chat_history = []
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def generation(vstore):
    retriever = vstore.as_retriever(search_kwargs={"k": 3})

    retriever_prompt = (
        "Given a chat history and the latest user question which might reference context in the chat history, "
        "formulate a standalone question which can be understood without the chat history. "
        "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", retriever_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)

    PRODUCT_BOT_TEMPLATE = """
    Your ecommercebot bot is an expert in product recommendations and customer queries.
    It analyzes product titles and reviews to provide accurate and helpful responses.
    Ensure your answers are relevant to the product context and refrain from straying off-topic.
    Your responses should be concise and informative.

    CONTEXT:
    {context}

    QUESTION: {input}

    YOUR ANSWER:
    """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PRODUCT_BOT_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ]
    )

    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain

if __name__ == "__main__":
    vstore = data_ingestion("done")
    conversational_rag_chain = generation(vstore)

    answer = conversational_rag_chain.invoke(
        {"input": "can you tell me the best bluetooth buds?"},
        config={"configurable": {"session_id": "ayush"}},
    )["answer"]
    print(answer)

    answer1 = conversational_rag_chain.invoke(
        {"input": "what is my previous question?"},
        config={"configurable": {"session_id": "ayush"}},
    )["answer"]
    print(answer1)