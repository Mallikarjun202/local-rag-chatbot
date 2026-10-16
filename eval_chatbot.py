"""
Test + efficiency check for the Chat_Bot server.

1. Start the server in one VS Code terminal (from the Chat_Bot folder):
       venv\\Scripts\\python.exe -m uvicorn app.main:app --port 8000
2. Run this in a second terminal:
       venv\\Scripts\\python.exe eval_chatbot.py
   Optional:  --model llama-3.1-8b-instant   (test only one model)
              --skip-llm                      (retrieval only, no API calls / no cost)

Uses only the Python standard library. Writes eval_results.json next to this file.
"""
import argparse
import json
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

BASE = "http://127.0.0.1:8000"

# Labeled retrieval set for the "default" collection (python_question_bank.pdf).
# "expect" = a phrase that appears in the chunk that SHOULD be retrieved.
# Questions are paraphrased on purpose so this isn't just keyword copy-paste.
RETRIEVAL_SET = [
    ("When should I use the := operator in Python?", "walrus operator"),
    ("Why does counting 'an' in banana return 2?", "'banana'.count('an')"),
    ("Why is multiplying a list containing an empty list by 3 a trap?", "[[]] * 3"),
    ("What goes wrong with a list as a default argument value?", "b=[]"),
    ("What does __slots__ prevent in a class?", "__slots__"),
    ("When is yield from better than plain yield?", "yield from"),
    ("What happens if I call blocking code inside an async coroutine?", "blocking function inside an async"),
    ("Why use a pytest fixture instead of setUp and tearDown?", "pytest fixture"),
    ("Common mistake with itertools groupby?", "itertools.groupby()"),
    ("Can you pickle a lambda or a closure?", "pickle a lambda"),
    ("Difference between Lock and RLock in threading?", "threading.RLock"),
    ("Time complexity of membership test in list versus set?", "'x in list' vs 'x in set'"),
]
# Questions the document can't answer: retrieval should return LOW scores.
OFF_TOPIC = [
    "What is the capital of France?",
    "How do I bake sourdough bread?",
]


def call(method, path, body=None, params=None, timeout=60):
    url = BASE + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read() or b"null")
            return out, (time.perf_counter() - t0) * 1000, None
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="ignore")[:300]
        return None, (time.perf_counter() - t0) * 1000, f"HTTP {e.code}: {detail}"
    except Exception as e:
        return None, (time.perf_counter() - t0) * 1000, f"{type(e).__name__}: {e}"


def pct(values, p):
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, round(p / 100 * (len(s) - 1))))
    return round(s[k], 1)


def line(ok, text):
    print(("  PASS  " if ok else "  FAIL  ") + text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="test only this model id")
    ap.add_argument("--skip-llm", action="store_true", help="no LLM calls")
    ap.add_argument("--collection", default="default")
    args = ap.parse_args()
    results = {"failures": []}

    # ---- 1. Server up --------------------------------------------------
    print("\n[1] Server")
    health, ms, err = call("GET", "/health")
    line(health == {"status": "ok"}, f"/health ({ms:.0f} ms) {err or ''}")
    if err:
        print("\nServer isn't reachable. Start it first (see top of this file).")
        return
    models, _, err = call("GET", "/models")
    available = [m["id"] for m in (models or {}).get("models", [])]
    line(bool(available), f"models with a key set: {available or (models or {}).get('message', err)}")

    # ---- 2. Retrieval quality + speed (no LLM, free) -------------------
    print(f"\n[2] Retrieval on collection '{args.collection}'")
    stats, _, err = call("GET", f"/rag/collections/{args.collection}")
    print(f"  {stats if stats else err}")
    hits1 = hits5 = 0
    rr, lat = [], []
    for q, expect in RETRIEVAL_SET:
        res, ms, err = call("POST", "/rag/search",
                            params={"collection": args.collection, "query": q, "top_k": 5})
        lat.append(ms)
        if err:
            results["failures"].append(f"search '{q}': {err}")
            line(False, f"{q} -> {err}")
            continue
        texts = [r["text"] for r in res["results"]]
        rank = next((i + 1 for i, t in enumerate(texts) if expect in t), None)
        hits1 += rank == 1
        hits5 += rank is not None
        rr.append(1 / rank if rank else 0)
        line(rank is not None, f"rank={rank or '-'}  {q}")
    n = len(RETRIEVAL_SET)
    off = []
    for q in OFF_TOPIC:
        res, ms, _ = call("POST", "/rag/search",
                          params={"collection": args.collection, "query": q, "top_k": 5})
        lat.append(ms)
        top = res["results"][0]["score"] if res and res["results"] else 0
        zero = sum(1 for r in (res or {}).get("results", []) if r["score"] <= 0)
        off.append({"q": q, "top_score": round(top, 3), "zero_score_results": zero})
    results["retrieval"] = {
        "questions": n, "hit@1": f"{hits1}/{n}", "hit@5": f"{hits5}/{n}",
        "hit@5_pct": round(100 * hits5 / n, 1), "MRR": round(sum(rr) / n, 3),
        "search_ms_p50": pct(lat, 50), "search_ms_p95": pct(lat, 95),
        "off_topic": off,
    }
    print(f"  hit@1 {hits1}/{n} | hit@5 {hits5}/{n} | MRR {results['retrieval']['MRR']}"
          f" | search p50 {pct(lat, 50)} ms, p95 {pct(lat, 95)} ms")
    for o in off:
        print(f"  off-topic top score {o['top_score']} ({o['zero_score_results']} of 5 results scored 0)  {o['q']}")

    # ---- 3. Each model: does the API call work, how fast? --------------
    to_test = [args.model] if args.model else available
    if args.skip_llm or not to_test:
        print("\n[3-5] Skipped LLM tests.")
    else:
        print("\n[3] Each model answers")
        results["models"] = {}
        for m in to_test:
            r, ms, err = call("POST", "/chat", {"session_id": f"eval-{uuid.uuid4()}",
                                                "message": "Reply with the single word: pong",
                                                "model": m, "use_memory": False})
            ok = bool(r and r.get("reply"))
            answered_by = (r or {}).get("model")
            note = (r or {}).get("fallback_note")
            results["models"][m] = {"ok": ok, "ms": round(ms), "error": err,
                                    "answered_by": answered_by, "note": note,
                                    "reply": (r or {}).get("reply", "")[:60]}
            if not ok:
                results["failures"].append(f"model {m}: {err or 'empty reply'}")
            extra = f"  [{note}]" if note else ""
            line(ok and answered_by == m, f"{m:28s} {ms:6.0f} ms  {err or repr((r or {}).get('reply', ''))[:40]}{extra}")

        working = [m for m, v in results["models"].items() if v["ok"] and v["answered_by"] == m]
        if working:
            m = working[0]
            # ---- 4. Memory ------------------------------------------------
            print(f"\n[4] Memory (using {m})")
            sid = f"eval-{uuid.uuid4()}"
            call("POST", "/chat", {"session_id": sid, "message": "My test codeword is ZEBRA42. Just say ok.",
                                   "model": m, "use_memory": True})
            r, _, err = call("POST", "/chat", {"session_id": sid, "message": "What is my test codeword?",
                                               "model": m, "use_memory": True})
            ok = bool(r and "ZEBRA42" in r["reply"].upper())
            line(ok, f"remembers across turns: {err or r['reply'][:80]!r}")
            call("DELETE", f"/session/{sid}/memory")
            r, _, _ = call("POST", "/chat", {"session_id": sid, "message": "What is my test codeword?",
                                             "model": m, "use_memory": True})
            forgot = bool(r and "ZEBRA42" not in r["reply"].upper())
            line(forgot, "forgets after clearing memory")
            results["memory"] = {"remembers": ok, "forgets_after_clear": forgot}

            # ---- 5. RAG chat end to end ------------------------------------
            print(f"\n[5] RAG chat (using {m})")
            rag_lat = []
            for q, _exp in RETRIEVAL_SET[:3]:
                r, ms, err = call("POST", "/chat", {"session_id": f"eval-{uuid.uuid4()}", "message": q,
                                                    "model": m, "use_memory": False, "use_rag": True,
                                                    "collection": args.collection})
                rag_lat.append(ms)
                note = (r or {}).get("fallback_note") or ""
                line(bool(r), f"{ms:6.0f} ms  sources={(r or {}).get('sources')}  {err or note}")
            results["rag_chat_ms"] = {"p50": pct(rag_lat, 50), "max": round(max(rag_lat))}

    # ---- Summary --------------------------------------------------------
    out = Path(__file__).with_name("eval_results.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nFailures: {len(results['failures'])}")
    for f in results["failures"]:
        print("  - " + f)
    print(f"Saved {out.name}")


if __name__ == "__main__":
    main()
