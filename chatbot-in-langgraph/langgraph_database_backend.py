from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import sqlite3

load_dotenv()

llm = ChatOpenAI()

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

#First we need to create a connection to the SQLite database, if db doesn't exist it will be created. We will use the sqlite3 library to do this. We will also create a checkpointer that will save the state of the chatbot to the database.

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)

#we are using check_same_thread=False because we will be using multiple threads, so it will stop checking if convo is same thread as of db or not.

# Checkpointer
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        #checkpointer.list(None) returns a list of all checkpoints in the database, we will iterate through them and add the thread_id to the set of all_threads.
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)


