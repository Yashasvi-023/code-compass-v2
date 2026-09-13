"""
app.py — Code Compass: GitHub RAG Chatbot
  • Login / Sign-up
  • ChatGPT-style sidebar with chat history (stored in MongoDB)
  • Each repository load = a new chat session
"""
import streamlit as st
from datetime import datetime

from auth import signup, login
from database import (
    create_chat, get_user_chats,
    get_chat_messages, get_chat,
    append_message
)
from model import GitHubRAGModel

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Code Compass",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="auto",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — dark theme, ChatGPT-style feel
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Global ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* ── Hide default Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }

  /* ── Floating hamburger toggle (replaces Streamlit's hidden arrow) ── */
  #sidebar-toggle-btn {
    position: fixed;
    top: 14px;
    left: 14px;
    z-index: 999999;
    background: #1a1a2e;
    border: 1px solid #3a3a6a;
    border-radius: 8px;
    width: 38px;
    height: 38px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    font-size: 18px;
    color: #a0aaff;
    transition: background 0.2s, border-color 0.2s;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
  }
  #sidebar-toggle-btn:hover {
    background: #252548;
    border-color: #4f6ef7;
  }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: #0f0f11;
    border-right: 1px solid #2a2a2e;
  }
  section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

  /* ── Chat bubbles ── */
  .user-bubble {
    background: #1e1e2e;
    border: 1px solid #3a3a5c;
    border-radius: 12px 12px 2px 12px;
    padding: 12px 16px;
    margin: 6px 0;
    max-width: 80%;
    margin-left: auto;
    color: #e0e0e0;
  }
  .assistant-bubble {
    background: #141420;
    border: 1px solid #2a2a40;
    border-radius: 12px 12px 12px 2px;
    padding: 12px 16px;
    margin: 6px 0;
    max-width: 85%;
    color: #e0e0e0;
  }

  /* ── Chat history item ── */
  .chat-item {
    padding: 8px 12px;
    border-radius: 8px;
    cursor: pointer;
    transition: background 0.15s;
    margin-bottom: 2px;
  }
  .chat-item:hover { background: #1e1e2e; }
  .chat-item.active { background: #1a1a3a; border-left: 3px solid #4f6ef7; }

  /* ── Auth card ── */
  .auth-card {
    max-width: 420px;
    margin: 60px auto 0 auto;
    background: #0f0f1a;
    border: 1px solid #2a2a40;
    border-radius: 16px;
    padding: 36px 32px;
  }
  .auth-title {
    font-size: 26px;
    font-weight: 700;
    color: #a0aaff;
    margin-bottom: 6px;
  }
  .auth-sub {
    font-size: 14px;
    color: #666;
    margin-bottom: 24px;
  }

  /* ── Gradient header ── */
  .brand {
    font-size: 20px;
    font-weight: 700;
    background: linear-gradient(90deg, #4f6ef7, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0px;
  }

  /* ── Sticker tags ── */
  .tag {
    display: inline-block;
    background: #1e1e3a;
    border: 1px solid #3a3a6a;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 11px;
    color: #9090cc;
    margin-right: 4px;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session-state initialisation
# ─────────────────────────────────────────────────────────────────────────────
defaults = {
    "user":           None,     # logged-in user dict
    "auth_mode":      "login",  # "login" | "signup"
    "current_chat_id": None,    # active chat _id (string)
    "rag_model":      None,     # GitHubRAGModel instance
    "messages":       [],       # messages displayed in the chat window
    "repo_loaded":    False,    # True when current chat has an active embedding
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# Helper: get API key
# ─────────────────────────────────────────────────────────────────────────────
def get_api_key() -> str:
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# AUTH PAGES
# ─────────────────────────────────────────────────────────────────────────────
def render_auth():
    # Centered brand header
    st.markdown(
        '<div style="text-align:center;padding:40px 0 10px 0;">'
        '<div class="brand">🧭 Code Compass</div>'
        '<div style="color:#555;font-size:13px;margin-top:4px;">Understand any GitHub repository with AI</div>'
        '</div>',
        unsafe_allow_html=True
    )

    col_l, col_m, col_r = st.columns([1, 1.6, 1])
    with col_m:
        # Tab toggle
        tab_login, tab_signup = st.tabs(["Log In", "Sign Up"])

        with tab_login:
            _render_login()

        with tab_signup:
            _render_signup()


def _render_login():
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    with st.form("login_form"):
        email    = st.text_input("Email", placeholder="you@example.com", key="li_email")
        password = st.text_input("Password", type="password", placeholder="••••••••", key="li_pass")
        submitted = st.form_submit_button("Log In", use_container_width=True)

    if submitted:
        if not email or not password:
            st.error("Please fill in all fields.")
        else:
            ok, msg, user = login(email, password)
            if ok:
                st.session_state.user = user
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)


def _render_signup():
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    with st.form("signup_form"):
        full_name = st.text_input("Full Name", placeholder="Alex Johnson", key="su_name")
        email     = st.text_input("Email",     placeholder="you@example.com", key="su_email")
        password  = st.text_input("Password",  type="password",
                                   placeholder="Min. 8 characters", key="su_pass")
        confirm   = st.text_input("Confirm Password", type="password",
                                   placeholder="Re-enter password", key="su_confirm")
        submitted = st.form_submit_button("Create Account", use_container_width=True)

    if submitted:
        if not all([full_name, email, password, confirm]):
            st.error("Please fill in all fields.")
        elif password != confirm:
            st.error("Passwords do not match.")
        else:
            ok, msg, user = signup(full_name, email, password)
            if ok:
                st.session_state.user = user
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP — SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    user = st.session_state.user

    with st.sidebar:
        # Brand
        st.markdown(f'<div class="brand">🧭 Code Compass</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="color:#555;font-size:12px;margin-bottom:16px;">'
            f'Signed in as <b style="color:#7a7aaa">{user["full_name"]}</b></div>',
            unsafe_allow_html=True
        )

        # New Chat button
        if st.button("＋  New Chat", use_container_width=True):
            st.session_state.current_chat_id = None
            st.session_state.rag_model       = None
            st.session_state.messages        = []
            st.session_state.repo_loaded     = False
            st.rerun()

        st.divider()

        # Chat history
        chats = get_user_chats(str(user["_id"]))

        if not chats:
            st.markdown('<div style="color:#444;font-size:13px;">No chats yet.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-size:11px;color:#555;margin-bottom:6px;">RECENT CHATS</div>',
                        unsafe_allow_html=True)
            for chat in chats:
                chat_id   = str(chat["_id"])
                is_active = (chat_id == st.session_state.current_chat_id)
                date_str  = chat["created_at"].strftime("%b %d")
                label     = f"{'📂' if is_active else '💬'}  {chat['repo_name']}"

                if st.button(
                    label,
                    key=f"chat_{chat_id}",
                    use_container_width=True,
                    help=f"{chat['repo_url']} · {date_str}",
                ):
                    _load_existing_chat(chat_id, chat)
                    st.rerun()

        st.divider()

        # Sign out
        if st.button("Sign Out", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


def _load_existing_chat(chat_id: str, chat: dict):
    """Switch the app to display an old chat (read-only; user can reload repo to continue)."""
    st.session_state.current_chat_id = chat_id
    st.session_state.repo_loaded     = False   # embeddings not in memory
    st.session_state.rag_model       = None

    # Load messages from MongoDB
    msgs = get_chat_messages(chat_id)
    st.session_state.messages = [
        {"role": m["role"], "content": m["content"]}
        for m in msgs
    ]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP — CHAT AREA
# ─────────────────────────────────────────────────────────────────────────────
def render_chat():
    user = st.session_state.user

    # ── Top bar ──────────────────────────────────────────────────────────────
    if st.session_state.current_chat_id:
        chat = get_chat(st.session_state.current_chat_id)
        if chat:
            st.markdown(
                f'<h3 style="margin:0;color:#a0aaff">📂 {chat["repo_name"]}</h3>'
                f'<div style="color:#555;font-size:12px;">{chat["repo_url"]}</div>',
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            '<h3 style="margin:0;color:#a0aaff">🧭 Code Compass</h3>'
            '<div style="color:#555;font-size:12px;">Load a GitHub repository to start chatting.</div>',
            unsafe_allow_html=True
        )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Repo loader (always visible when not loaded) ──────────────────────────
    if not st.session_state.repo_loaded:
        with st.expander("📥 Load a GitHub Repository", expanded=(not st.session_state.current_chat_id)):
            repo_url = st.text_input(
                "GitHub Repository URL",
                placeholder="https://github.com/username/repo",
                key="repo_url_input"
            )
            if st.button("Load Repository", type="primary"):
                api_key = get_api_key()
                if not api_key:
                    st.error("Gemini API key not found in secrets.")
                elif not repo_url:
                    st.error("Please enter a repository URL.")
                else:
                    with st.spinner("Cloning and indexing repository… this may take a minute."):
                        try:
                            model = GitHubRAGModel(api_key)
                            ok, msg = model.process_repository(repo_url)
                        except Exception as e:
                            ok, msg = False, str(e)

                    if ok:
                        # Create a new chat in MongoDB
                        repo_name = repo_url.rstrip("/").split("/")[-1] \
                                    .replace("-", " ").replace("_", " ")
                        chat_id = create_chat(
                            user_id=str(user["_id"]),
                            repo_url=repo_url,
                            repo_name=repo_name,
                        )
                        st.session_state.current_chat_id = chat_id
                        st.session_state.rag_model       = model
                        st.session_state.repo_loaded     = True
                        st.session_state.messages        = []
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

    # If user is viewing an old chat without embeddings, offer a reload option
    if st.session_state.current_chat_id and not st.session_state.repo_loaded:
        chat = get_chat(st.session_state.current_chat_id)
        if chat:
            st.info(
                f"💡 You're viewing a past chat. To ask new questions about "
                f"**{chat['repo_name']}**, reload the repository above."
            )

    st.divider()

    # ── Message history ───────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Chat input ────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask about the codebase…"):
        if not st.session_state.repo_loaded:
            st.warning("Please load a repository first.")
        else:
            # Display user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Persist user message
            append_message(st.session_state.current_chat_id, "user", prompt)

            # Generate answer
            with st.chat_message("assistant"):
                with st.spinner("Searching codebase…"):
                    try:
                        answer = st.session_state.rag_model.ask_question(prompt)
                    except Exception as e:
                        answer = f"⚠️ Error: {e}"

                st.markdown(answer)

            # Persist assistant message
            st.session_state.messages.append({"role": "assistant", "content": answer})
            append_message(st.session_state.current_chat_id, "assistant", answer)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.user is None:
    render_auth()
else:
    render_sidebar()
    render_chat()
