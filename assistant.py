"""
assistant.py
============
Stage 5. Conversational layer -- GroqCloud, not Gemini, not xAI's "Grok".

Why GroqCloud specifically:
  - Genuine ongoing free tier: no credit card, no data-sharing opt-in
    required. ~30 RPM / 1,000 RPD / 12K TPM per model as of this build
    (June 2026) -- verify current numbers at console.groq.com, free-tier
    limits do get revised. This is different from xAI's Grok API, which
    is pay-per-token with only a time-limited signup credit -- if you
    actually meant xAI, this file needs different code (xAI's API is
    also OpenAI-compatible, so the swap is small, but the cost math
    changes back to "not free").
  - OpenAI-compatible `tools`/`tool_calls` format -- standard function
    calling, well documented, same shape as most agent frameworks.
  - Runs open-weight models (Llama, Qwen, GPT-OSS, Kimi) on custom LPU
    hardware -- fast, no proprietary-model lock-in.

Model: llama-3.3-70b-versatile (quality pick for reasoning + explanation
text). Swap MODEL to "llama-3.1-8b-instant" if you hit free-tier rate
limits during a live demo -- much higher RPM, lower quality ceiling but
plenty for this catalog size.

GroqCloud does NOT auto-execute Python functions the way Gemini's SDK
did -- this implements the standard manual tool-call loop: send the
schema, get back tool_calls, run them yourself, send the result back as
a "tool" role message, call again for the final answer.

NOT TESTED LIVE -- no network egress to api.groq.com in this build
sandbox. retrieve_outfit() itself (everything except the actual Groq
call) IS tested, since it's calling your already-tested retrieval.py.

Run:
    export GROQ_API_KEY="your-key-from-console.groq.com"
    python assistant.py --processed_dir ./processed --data_dir .

Requires: groq  (pip install groq)
"""

import argparse
import json
import os

import pandas as pd

import retrieval

DEFAULT_MODEL = "llama-3.3-70b-versatile"
MAX_TOOL_ROUNDS = 4

SYSTEM_PROMPT = """You are a fashion outfit assistant for an Indian fashion catalog.

Your job:
1. Figure out the user's gender (men/women), occasion (e.g. office, party,
   casual, wedding, festive, sports, vacation, winter), and a short style
   description from their message.
2. If gender OR occasion is missing and you cannot reasonably infer it from
   the conversation, ask ONE short clarifying question instead of calling
   any tool. Do not guess gender.
3. Once you have gender + occasion (style description can be a short
   paraphrase of whatever the user said), call retrieve_outfit with:
     - gender: "men" or "women"
     - occasion: one of the values above (pick the closest match)
     - style_text: a short free-text description capturing what they want
     - anchor_item: leave empty unless the user names a specific item
4. retrieve_outfit returns up to 3 outfit options, each with real item
   names, categories, prices, a feature breakdown, and a "style_reference"
   (a real stylist's note on a similar curated outfit -- ground your
   reasoning in this, PARAPHRASE it in your own words, never quote it
   verbatim, never invent reasoning that contradicts it).
5. Present each outfit as: the list of items (name + category), then 1-2
   sentences explaining WHY they work together for the stated occasion --
   reference actual attributes (color, category, occasion) you were given,
   not generic fashion platitudes.
6. NEVER invent an item that wasn't returned by retrieve_outfit. If
   retrieve_outfit returns zero outfits, say so plainly and suggest
   loosening the occasion or trying a different style.
"""

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "retrieve_outfit",
        "description": (
            "Finds 1-3 complete outfit recommendations from the catalog, "
            "each with real item names, categories, prices, a compatibility "
            "feature breakdown, and a grounding style_reference."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "gender": {"type": "string", "enum": ["men", "women"]},
                "occasion": {
                    "type": "string",
                    "description": (
                        "e.g. office, party, casual, wedding, festive, "
                        "sports, vacation, winter"
                    ),
                },
                "style_text": {
                    "type": "string",
                    "description": "Short free-text description of what the user wants.",
                },
                "anchor_item": {
                    "type": "string",
                    "description": "Optional specific product id to build the outfit around.",
                },
            },
            "required": ["gender", "occasion", "style_text"],
        },
    },
}


# ---------------------------------------------------------------------------
# Lazy-loaded pipeline artifacts (shared across all retrieve_outfit calls)
# ---------------------------------------------------------------------------
_ARTIFACTS = {}


def _load_artifacts_once(processed_dir: str, data_dir: str):
    if _ARTIFACTS:
        return _ARTIFACTS
    products, featurizer, clf, minilm_emb, id_to_idx = retrieval.load_artifacts(processed_dir)
    with open(os.path.join(processed_dir, "outfits_parsed.json")) as f:
        parsed_outfits = json.load(f)
    _ARTIFACTS.update(
        products=products, featurizer=featurizer, clf=clf,
        minilm_emb=minilm_emb, id_to_idx=id_to_idx, parsed_outfits=parsed_outfits,
        data_dir=data_dir,
    )
    return _ARTIFACTS


def _nearest_ground_truth_outfit(item_ids: list, gender: str, parsed_outfits: list):
    best, best_overlap = None, -1
    item_set = set(item_ids)
    for outfit in parsed_outfits:
        if outfit["gender"] != gender:
            continue
        outfit_ids = {it["id"] for it in outfit["items"]}
        overlap = len(item_set & outfit_ids)
        if overlap > best_overlap:
            best_overlap, best = overlap, outfit
    return best


def _build_retrieve_outfit(processed_dir: str, data_dir: str, capture: list | None = None):
    """Returns a retrieve_outfit() closure bound to the loaded artifacts --
    this is what the manual tool-call loop executes when Groq asks for it."""

    def retrieve_outfit(gender: str, occasion: str, style_text: str, anchor_item: str = "") -> dict:
        art = _load_artifacts_once(processed_dir, data_dir)
        query = {
            "gender": gender,
            "occasion": occasion,
            "style_text": style_text,
            "anchor_item": anchor_item or None,
        }
        combos = retrieval.recommend(
            query, art["products"], art["featurizer"], art["clf"],
            art["id_to_idx"], art["minilm_emb"], top_n=3,
        )

        rows = art["products"].set_index("id")
        outfits_out = []
        for combo in combos:
            items_out = []
            for item_id in combo["items"]:
                r = rows.loc[item_id]
                items_out.append({
                    "id": item_id,
                    "name": r["name"],
                    "category": r["category_label"],
                    "price_inr": int(r["price_inr"]) if pd.notna(r["price_inr"]) else None,
                    "image": r["image"],
                })
            ref = _nearest_ground_truth_outfit(combo["items"], gender, art["parsed_outfits"])
            outfits_out.append({
                "items": items_out,
                "compat_score": round(combo["compat_avg"], 3),
                "relevance_score": round(combo["relevance_avg"], 3),
                "feature_breakdown": combo.get("feature_breakdown"),
                "style_reference": {
                    "theme": ref["theme"] if ref else None,
                    "palette": ref["palette"] if ref else None,
                    "stylist_rationale": ref["stylist_rationale"] if ref else None,
                } if ref else None,
            })

        result = {"outfits": outfits_out}
        if capture is not None:
            capture.append(result)
        return result

    return retrieve_outfit


# ---------------------------------------------------------------------------
# Chat session -- manual OpenAI-style tool-call loop
# ---------------------------------------------------------------------------
class GroqAssistantSession:
    def __init__(self, processed_dir: str, data_dir: str,
                api_key: str | None = None, model: str = DEFAULT_MODEL):
        from groq import Groq

        self.client = Groq(api_key=api_key)  # falls back to GROQ_API_KEY env var
        self.model = model
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.capture: list = []
        self._retrieve_outfit = _build_retrieve_outfit(processed_dir, data_dir, capture=self.capture)

    def send_message(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})

        for _ in range(MAX_TOOL_ROUNDS):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=[TOOL_SCHEMA],
                tool_choice="auto",
            )
            msg = response.choices[0].message
            self.messages.append(msg.model_dump(exclude_none=True))

            if not msg.tool_calls:
                return msg.content or ""

            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                    result = self._retrieve_outfit(**args)
                except Exception as e:  # surfaces a bad tool call instead of crashing the session
                    result = {"error": str(e)}
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })

        return "Sorry, I'm having trouble completing that request right now -- try rephrasing."


def build_chat(processed_dir: str, data_dir: str, api_key: str | None = None,
                model: str = DEFAULT_MODEL) -> GroqAssistantSession:
    return GroqAssistantSession(processed_dir, data_dir, api_key=api_key, model=model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="./processed")
    ap.add_argument("--data_dir", default=".")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        print("[!] GROQ_API_KEY not set. export GROQ_API_KEY=... and re-run.")
        return

    session = build_chat(args.processed_dir, args.data_dir, model=args.model)
    print(f"Fashion assistant ready (model={args.model}). Type 'quit' to exit.\n")
    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in ("quit", "exit"):
            break
        session.capture.clear()
        reply = session.send_message(user_text)
        print(f"\nAssistant: {reply}\n")
        if session.capture:
            print(f"[{len(session.capture[-1]['outfits'])} structured outfit(s) also returned by the tool call]\n")


if __name__ == "__main__":
    main()
