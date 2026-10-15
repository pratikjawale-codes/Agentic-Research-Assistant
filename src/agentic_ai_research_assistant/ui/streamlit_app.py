import asyncio

import streamlit as st

from agentic_ai_research_assistant.agents.graph import research_graph
from agentic_ai_research_assistant.config import settings
from agentic_ai_research_assistant.database.session import async_engine, init_db
from agentic_ai_research_assistant.knowledge_base.bm25 import global_bm25_manager
from agentic_ai_research_assistant.knowledge_base.chunker import create_chunks
from agentic_ai_research_assistant.knowledge_base.parser import parse_document
from agentic_ai_research_assistant.knowledge_base.vector_store import global_vector_store
from agentic_ai_research_assistant.memory.long_term import LongTermMemoryManager

st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink: #14243a; --muted: #64748b; --brand: #3157d5; }
    .stApp { background: #f5f7fb; color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: #fff; border-right: 1px solid #e8edf5; }
    .block-container { max-width: 1240px; padding-top: 2.3rem; padding-bottom: 3rem; }
    .hero {
        padding: 2.2rem 2.5rem; border-radius: 24px; margin-bottom: 1.6rem;
        color: white; background: linear-gradient(120deg, #14243a 0%, #2447a8 65%, #5875e8 100%);
        box-shadow: 0 16px 40px rgba(36, 71, 168, .18);
    }
    .hero-kicker { color: #c8d6ff; font-size: .78rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }
    .hero h1 { margin: .55rem 0; color: white; font-size: clamp(2rem, 4vw, 3rem); letter-spacing: -.04em; }
    .hero p { margin: 0; max-width: 760px; color: #e2eaff; font-size: 1.04rem; line-height: 1.65; }
    [data-testid="stMetric"] { padding: 1rem; border: 1px solid #e7ebf2; border-radius: 16px; background: white; }
    [data-testid="stVerticalBlockBorderWrapper"] { border-color: #e7ebf2; border-radius: 18px; }
    .stButton > button[kind="primary"] { min-height: 3rem; border-radius: 11px; font-weight: 650; }
    .stTextArea textarea { border-radius: 12px; }
    h2, h3 { letter-spacing: -.025em; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def initialize_workspace() -> bool:
    asyncio.run(init_db())
    asyncio.run(async_engine.dispose())
    return True


initialize_workspace()

provider_ready = any(
    (settings.OPENAI_API_KEY, settings.GEMINI_API_KEY, settings.ANTHROPIC_API_KEY)
)

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Your research workspace</div>
      <h1>Turn complex questions into clear answers.</h1>
      <p>Bring your documents, market intelligence, and verified calculations together in one grounded research workflow.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Workspace")
    workspace = st.selectbox(
        "Choose a workspace",
        options=["Enterprise Workspace A · 101", "Enterprise Workspace B · 102"],
        label_visibility="collapsed",
    )
    user_id = 101 if "101" in workspace else 102
    st.caption(f"Private documents and saved research are scoped to workspace **{user_id}**.")

    st.divider()
    st.markdown("### Knowledge library")
    st.caption("Add source material to ground reports in your own documents.")
    uploaded_file = st.file_uploader(
        "PDF, DOCX, TXT, or Markdown",
        type=["pdf", "docx", "txt", "md"],
        label_visibility="collapsed",
    )
    if uploaded_file and st.button("Add to knowledge library", use_container_width=True):
        try:
            with st.spinner("Reading and indexing your document…"):
                pages = parse_document(uploaded_file.getvalue(), uploaded_file.name)
                doc_id = abs(hash((user_id, uploaded_file.name))) % 10_000
                chunks = create_chunks(
                    pages_data=pages,
                    user_id=user_id,
                    doc_id=doc_id,
                    filename=uploaded_file.name,
                    chunk_size=settings.RAG_CHUNK_SIZE,
                    chunk_overlap=settings.RAG_CHUNK_OVERLAP,
                )
                global_vector_store.add_chunks(chunks)
                all_user_chunks = [
                    chunk for chunk in global_vector_store.chunks.values()
                    if chunk["user_id"] == user_id
                ]
                global_bm25_manager.invalidate(user_id)
                global_bm25_manager.get_or_build_index(user_id, all_user_chunks)
            st.success(f"Added {len(chunks)} searchable sections.")
        except Exception as error:
            st.error(f"Could not index this document: {error}")
            st.exception(error)

    st.divider()
    st.markdown("### Saved research")
    try:
        memories = LongTermMemoryManager.get_user_memories(user_id)
        if memories:
            for memory in memories:
                st.markdown(f"**{memory['type'].replace('_', ' ').title()}**")
                st.caption(memory["content"][:140])
        else:
            st.caption("Your completed research will appear here.")
    except Exception as error:
        st.error(f"Saved research is unavailable: {error}")

    st.divider()
    if provider_ready:
        st.success("AI provider configured")
    else:
        st.info("Local mode is ready. Add a provider key to `.env` whenever you are ready.")

st.markdown("### Start a research task")
task_column, workflow_column = st.columns([1.35, 0.85], gap="large")
with task_column:
    query = st.text_area(
        "What would you like to investigate?",
        height=150,
        placeholder=(
            "For example: Compare the growth outlook for enterprise software, "
            "check the pricing assumptions in my uploaded report, and calculate a five-year CAGR."
        ),
        label_visibility="collapsed",
    )
    run_btn = st.button(
        "Run research",
        type="primary",
        icon=":material/auto_awesome:",
        use_container_width=True,
    )
with workflow_column:
    with st.container(border=True):
        st.markdown("#### One connected workflow")
        st.markdown(
            "- **Research** gathers document and public evidence.\n"
            "- **Analysis** calculates metrics with deterministic tools.\n"
            "- **Synthesis** turns findings into a cited report."
        )
        st.caption("Provider keys are optional; local tools and document search work without them.")

if run_btn:
    if not query.strip():
        st.warning("Enter a research question to get started.")
    else:
        initial_state = {
            "messages": [{"role": "user", "content": query.strip()}],
            "intent": {},
            "plan": [],
            "current_step": 0,
            "findings": {},
            "numbers": {},
            "report": "",
            "replans": 0,
            "errors": [],
        }
        config = {
            "configurable": {
                "user_id": str(user_id),
                "thread_id": f"{user_id}:chat_session",
            }
        }
        try:
            with st.spinner("Coordinating research, evidence, and analysis…"):
                output = research_graph.invoke(initial_state, config=config)
            st.success("Research complete")
            report_tab, evidence_tab, plan_tab = st.tabs(
                ["Research report", "Evidence & analysis", "Workflow"]
            )
            with report_tab:
                st.markdown(output.get("report", "No report was generated."))
            with evidence_tab:
                left, right = st.columns(2)
                with left:
                    st.markdown("#### Collected evidence")
                    st.json(output.get("findings", {}))
                with right:
                    st.markdown("#### Verified calculations")
                    st.json(output.get("numbers", {}))
            with plan_tab:
                st.markdown("#### Research plan")
                st.json(output.get("intent", {}))
                for step in output.get("plan", []):
                    st.markdown(
                        f"**{step['id'] + 1}. {step['agent'].title()}** — {step['description']}"
                    )
        except Exception as error:
            st.error(f"Research could not be completed: {error}")
            st.exception(error)

st.divider()
st.caption("Agentic AI Research Assistant · Built by Pratik Jawale")
