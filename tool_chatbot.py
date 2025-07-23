# ─────────────────────────────────────────────────────────────────────────────
# 1. Imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from typing_extensions import TypedDict
from typing import Annotated

# ─────────────────────────────────────────────────────────────────────────────
# 2. Build agent (fixed wrappers)
# ─────────────────────────────────────────────────────────────────────────────
groq_api_key=os.getenv("GROQ_API_KEY")

arxiv=ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(top_k_results=5, doc_content_chars_max=500_000)
)
wiki=WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=5, doc_content_chars_max=500_000)
)
tavily=TavilySearchResults(max_results=5)

tools=[arxiv, wiki, tavily]
llm=ChatGroq(model="qwen/qwen3-32b")
llm_with_bind=llm.bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def tool_calling_llm(state):
    return {"messages":[llm_with_bind.invoke(state["messages"])]}

builder=StateGraph(State)
builder.add_node("tool_calling_llm",tool_calling_llm)
builder.add_node("tools",ToolNode(tools))
builder.add_edge(START,"tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm",tools_condition)
builder.add_edge("tools",END)
graph=builder.compile()

# ─────────────────────────────────────────────────────────────────────────────
# 3. Theme & styling
# ─────────────────────────────────────────────────────────────────────────────
THEMES={
    "Blue":{
        "bg":"linear-gradient(135deg,#667eea 0%,#764ba2 100%)",
        "chat_bg":"rgba(255,255,255,0.10)",
        "user_text":"#ffffff",
        "bot_text":"#ffffff",
        "accent":"#667eea",
    },
    "Purple":{
        "bg":"linear-gradient(135deg,#f093fb 0%,#f5576c 100%)",
        "chat_bg":"rgba(255,255,255,0.12)",
        "user_text":"#ffffff",
        "bot_text":"#ffffff",
        "accent":"#f093fb",
    },
    "Green":{
        "bg":"linear-gradient(135deg,#4facfe 0%,#00f2fe 100%)",
        "chat_bg":"rgba(255,255,255,0.08)",
        "user_text":"#ffffff",
        "bot_text":"#ffffff",
        "accent":"#00f2fe",
    },
}

st.set_page_config(page_title="Agent", page_icon="🤖", layout="wide")
with st.sidebar:
    st.title("🎨 Theme")
    chosen=st.selectbox("Pick palette:", list(THEMES.keys()))
theme=THEMES[chosen]

st.markdown(
    f"""
    <style>
    .stApp {{
        background:{theme["bg"]};
        color:white;
        font-family:"Segoe UI",Helvetica,Arial,sans-serif;
    }}
    footer{{visibility:hidden}}
    [data-testid="stSidebar"]{{background:rgba(0,0,0,0.15);backdrop-filter:blur(3px)}}
    .chat-row{{display:flex;margin-bottom:1rem;align-items:flex-start}}
    .user-bubble,.bot-bubble{{
        max-width:75%;padding:0.8rem 1.2rem;border-radius:1.2rem;
        backdrop-filter:blur(8px);box-shadow:0 4px 15px rgba(0,0,0,.1);
        animation:fadeIn .4s ease-in-out;
    }}
    .user-bubble{{
        background:{theme["chat_bg"]};color:{theme["user_text"]};
        margin-left:auto;border-bottom-right-radius:0;
    }}
    .bot-bubble{{
        background:{theme["chat_bg"]};color:{theme["bot_text"]};
        border-bottom-left-radius:0;
    }}
    .icon{{font-size:2rem;margin:.3rem .5rem}}
    @keyframes fadeIn{{from{{opacity:0;transform:translateY(10px)}}to{{opacity:1;transform:translateY(0)}}}}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Session state
# ─────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ─────────────────────────────────────────────────────────────────────────────
# 5. Display chat history + Title
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;color:black;'>❤️‍🔥 Dil-Se-Agent</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center;color:black;'>Dil se poocho, Dil se jawab.</p>",
    unsafe_allow_html=True,
)

for msg in st.session_state.messages:
    side = "user-bubble" if msg["role"] == "user" else "bot-bubble"
    icon = "👤" if msg["role"] == "user" else "🤖"
    st.markdown(
        f"""
        <div class="chat-row">
            <div class="icon">{icon}</div>
            <div class="{side}">{msg["content"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# 6. Chat input & agent response
# ─────────────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    prompt = st.session_state.messages[-1]["content"]
    with st.spinner("🧠 Thinking …"):
        final_state = graph.invoke({"messages": [HumanMessage(content=prompt)]})

    # Build plain-text answer
    lines = []
    for m in final_state["messages"]:
        if getattr(m, "tool_calls", None):
            for call in m.tool_calls:
                tool_name = call["name"]
                args = ", ".join([f"{k}={v}" for k, v in call["args"].items()])
                lines.append(f"🔍 Used tool: {tool_name}\n   Arguments: {args}")
        if not isinstance(m, (HumanMessage, AIMessage)):  # ToolMessage
            lines.append(f"📦 Tool result:\n{m.content}")
        if isinstance(m, AIMessage) and m.content:
            lines.append(m.content)

    answer = "\n\n".join(lines) or " Agent finished (no extra text)."
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()