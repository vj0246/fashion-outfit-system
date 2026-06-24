"""
backend/main.py
================
Stage 6. FastAPI backend wrapping the conversational assistant.

Endpoints:
  GET  /health                          -> {"status": "ok"}
  POST /chat       {session_id, message} -> {reply, outfits}
  POST /recommend  {gender, occasion, style_text, anchor_item?} -> {outfits}
  GET  /graph                            -> {nodes, edges} (co-occurrence graph)

/chat   drives assistant.py (GroqCloud, full conversational flow,
        slot-filling, explainability). `outfits` is the structured data
        from the retrieve_outfit tool call this turn (empty list if the
        model asked a clarifying question instead of calling the tool).
/recommend bypasses the LLM entirely and calls retrieval.py directly --
        useful for a UI panel that needs deterministic structured results
        without waiting on/depending on the LLM (and works even before you
        have a Groq key wired up, for local dev).
/graph  returns the real item-item co-occurrence graph from
        data_pipeline.py, for the frontend's network visualization.

Run (from the project root, with backend/ as a subfolder):
    export GROQ_API_KEY="your-key"
    uvicorn backend.main:app --reload --port 8000

Then POST to http://localhost:8000/chat etc. Swagger UI at /docs.

Requires: fastapi, uvicorn[standard]  (pip install fastapi "uvicorn[standard]")
"""

import json
import os
import sys
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

# allow `import retrieval`, `import assistant` etc. from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import assistant
import retrieval

PROCESSED_DIR = os.environ.get("PROCESSED_DIR", "./processed")
DATA_DIR = os.environ.get("DATA_DIR", ".")

app = FastAPI(title="Fashion Outfit Recommendation API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # open for the Vercel static frontend; tighten to your
                          # actual vercel.app domain before sharing this publicly
    allow_methods=["*"],
    allow_headers=["*"],
)

# session_id -> GroqAssistantSession  -- in-memory, fine for an MVP demo.
# Restarting the server drops all sessions. Swap for redis/db if this needs
# to survive restarts or run multi-process.
_SESSIONS: dict = {}

# lazily-loaded direct-retrieval artifacts (for /recommend, no LLM needed)
_RETRIEVAL_ARTIFACTS: dict = {}


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    outfits: list


class RecommendRequest(BaseModel):
    gender: str
    occasion: Optional[str] = None
    style_text: str
    anchor_item: Optional[str] = None
    top_n: int = 3


@app.get("/health")
def health():
    return {"status": "ok"}


def _get_or_create_session(session_id: str) -> assistant.GroqAssistantSession:
    if session_id not in _SESSIONS:
        if not os.environ.get("GROQ_API_KEY"):
            raise HTTPException(
                status_code=500,
                detail="GROQ_API_KEY is not set on the server. export it and restart.",
            )
        _SESSIONS[session_id] = assistant.build_chat(PROCESSED_DIR, DATA_DIR)
    return _SESSIONS[session_id]


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    session = _get_or_create_session(req.session_id)
    session.capture.clear()
    try:
        reply = session.send_message(req.message)
    except Exception as e:  # surfaces real Groq errors (quota, bad key, etc.) to the client
        raise HTTPException(status_code=502, detail=f"Groq call failed: {e}")

    outfits = session.capture[-1]["outfits"] if session.capture else []
    return ChatResponse(reply=reply, outfits=outfits)


def _get_retrieval_artifacts():
    if not _RETRIEVAL_ARTIFACTS:
        products, featurizer, clf, minilm_emb, id_to_idx = retrieval.load_artifacts(PROCESSED_DIR)
        _RETRIEVAL_ARTIFACTS.update(
            products=products, featurizer=featurizer, clf=clf,
            minilm_emb=minilm_emb, id_to_idx=id_to_idx,
        )
    return _RETRIEVAL_ARTIFACTS


@app.post("/recommend")
def recommend_endpoint(req: RecommendRequest):
    art = _get_retrieval_artifacts()
    query = {
        "gender": req.gender,
        "occasion": req.occasion,
        "style_text": req.style_text,
        "anchor_item": req.anchor_item,
    }
    combos = retrieval.recommend(
        query, art["products"], art["featurizer"], art["clf"],
        art["id_to_idx"], art["minilm_emb"], top_n=req.top_n,
    )
    rows = art["products"].set_index("id")
    outfits = []
    for combo in combos:
        items = []
        for iid in combo["items"]:
            price = rows.loc[iid, "price_inr"]
            items.append({
                "id": iid,
                "name": rows.loc[iid, "name"],
                "category": rows.loc[iid, "category_label"],
                "price_inr": int(price) if pd.notna(price) else None,
                "image": rows.loc[iid, "image"],
            })
        outfits.append({
            "items": items,
            "compat_score": combo["compat_avg"],
            "relevance_score": combo["relevance_avg"],
            "feature_breakdown": combo.get("feature_breakdown"),
        })
    return {"outfits": outfits}


@app.get("/graph")
def graph_endpoint():
    """Real item-item co-occurrence graph from data_pipeline.py, for the
    frontend's network visualization (vis-network/D3 etc.)."""
    graph_path = os.path.join(PROCESSED_DIR, "cooccurrence_graph.json")
    if not os.path.exists(graph_path):
        raise HTTPException(status_code=404, detail="cooccurrence_graph.json not found -- run data_pipeline.py first.")
    with open(graph_path) as f:
        graph = json.load(f)

    products = pd.read_csv(os.path.join(PROCESSED_DIR, "products_with_slots.csv")).set_index("id")
    node_ids = sorted(set(graph.keys()) | {b for nbrs in graph.values() for b in nbrs})

    nodes = [
        {
            "id": pid,
            "label": products.loc[pid, "name"] if pid in products.index else pid,
            "slot": products.loc[pid, "slot"] if pid in products.index else "unknown",
            "gender": products.loc[pid, "gender"] if pid in products.index else "unknown",
        }
        for pid in node_ids
    ]
    seen_edges, edges = set(), []
    for a, neighbors in graph.items():
        for b, weight in neighbors.items():
            key = frozenset({a, b})
            if key in seen_edges:
                continue
            seen_edges.add(key)
            edges.append({"source": a, "target": b, "weight": weight})

    return {"nodes": nodes, "edges": edges}
