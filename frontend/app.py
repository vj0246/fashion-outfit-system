"""
frontend/app.py
================
Stage 7. Streamlit chat UI -- talks to the FastAPI backend over HTTP
(backend/main.py). Chosen over a React frontend deliberately: same
architecture value (decoupled API + client), ~1 hour to build instead of
3-4, and System Design is only 10% of the eval weight here -- the budget
is better spent on the ML stages.

Two panels:
  - Chat tab: natural-language conversation -> POST /chat (Groq-driven,
    needs GROQ_API_KEY set on the backend).
  - Direct Search tab: structured gender/occasion/style form -> POST
    /recommend (no LLM, works even without a Groq key -- useful for
    demoing the retrieval+compatibility engine on its own).

Run:
    # terminal 1
    export GROQ_API_KEY="your-key"
    uvicorn backend.main:app --reload --port 8000

    # terminal 2
    streamlit run frontend/app.py

Requires: streamlit, requests  (pip install streamlit requests)
"""

import os
import uuid

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
DATA_DIR = os.environ.get("DATA_DIR", ".")  # root folder containing images/

st.set_page_config(page_title="AI Fashion Outfit Assistant", page_icon="👗", layout="wide")


def render_outfits(outfits: list):
    if not outfits:
        return
    for i, outfit in enumerate(outfits, 1):
        with st.container(border=True):
            st.markdown(f"**Outfit {i}**  ·  compat={outfit.get('compat_score', 0):.2f}  "
                        f"·  relevance={outfit.get('relevance_score', 0):.2f}")
            cols = st.columns(len(outfit["items"]))
            for col, item in zip(cols, outfit["items"]):
                with col:
                    img_path = os.path.join(DATA_DIR, item["image"])
                    if os.path.exists(img_path):
                        st.image(img_path, use_container_width=True)
                    else:
                        st.caption("(image not found)")
                    st.caption(f"**{item['name']}**\n\n{item['category']}")
                    if item.get("price_inr") is not None:
                        st.caption(f"₹{item['price_inr']}")
            ref = outfit.get("style_reference")
            if ref and ref.get("stylist_rationale"):
                st.markdown(f"_Style reference ({ref['theme']}): {ref['stylist_rationale']}_")


def backend_alive() -> bool:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=2)
        return r.status_code == 200
    except requests.RequestException:
        return False


st.title("👗 AI Fashion Outfit Assistant")

if not backend_alive():
    st.error(
        f"Backend not reachable at {BACKEND_URL}. Start it first:\n\n"
        f"`uvicorn backend.main:app --reload --port 8000`"
    )

tab_chat, tab_search = st.tabs(["💬 Chat", "🔍 Direct Search"])

# ---------------------------------------------------------------------------
# Chat tab (Groq-driven, slot-filling, explainability)
# ---------------------------------------------------------------------------
with tab_chat:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("outfits"):
                render_outfits(msg["outfits"])

    user_input = st.chat_input("e.g. 'I need an outfit for a business meeting'")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/chat",
                        json={"session_id": st.session_state.session_id, "message": user_input},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    st.write(data["reply"])
                    render_outfits(data["outfits"])
                    st.session_state.messages.append(
                        {"role": "assistant", "content": data["reply"], "outfits": data["outfits"]}
                    )
                except requests.RequestException as e:
                    err = f"Request failed: {e}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err, "outfits": []})

# ---------------------------------------------------------------------------
# Direct search tab (bypasses the LLM entirely -- pure retrieval+compatibility demo)
# ---------------------------------------------------------------------------
with tab_search:
    st.caption("Bypasses the LLM entirely -- hits /recommend directly. Works without a Groq key.")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["women", "men"])
        occasion = st.selectbox(
            "Occasion",
            ["casual", "party", "office", "festive", "wedding", "sports", "vacation", "winter"],
        )
    with col2:
        style_text = st.text_input("Style description", "smart casual outfit")
        anchor_item = st.text_input("Anchor item id (optional)", "")

    if st.button("Find outfits", type="primary"):
        with st.spinner("Searching..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/recommend",
                    json={
                        "gender": gender,
                        "occasion": occasion,
                        "style_text": style_text,
                        "anchor_item": anchor_item or None,
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                render_outfits(resp.json()["outfits"])
            except requests.RequestException as e:
                st.error(f"Request failed: {e}")
