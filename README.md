# Local RAG Chatbot

A small chatbot I run on my own laptop. It can talk to Groq, Gemini, OpenAI or Claude, remembers the conversation, and can answer questions from PDFs or text files you upload.

I built it to learn how retrieval-augmented generation actually behaves outside a tutorial: what happens when a provider is overloaded, when a question has nothing to do with your documents, and how slow "free tier" really is.

## What it does

- Chat with any provider you have a key for. Models without a key (or with a placeholder like `put-your-key-here`) don't show up in the list.
- Upload a `.pdf`, `.txt` or `.md` file and turn on "Use documents". The text is split into chunks and searched with TF-IDF, so it works offline and costs nothing. OpenAI embeddings are optional.
- Questions that don't match your documents (a score below 0.05) are answered without document context, instead of stuffing random chunks into the prompt.
- If a model returns 503 or 429, the app waits 1.5 s and retries once, then switches to a model from a different provider. Every reply shows which model answered and how long it took.
- A "Think" button asks the model to reason more before answering, and shows the reasoning when the provider sends it back.
- Per-conversation memory you can turn off or clear.

## Running it

You need Python 3.13 or newer (I've tested on 3.13 and 3.14) and at least one API key. Groq and Google AI Studio both have free tiers.

```bash
git clone https://github.com/Mallikarjun202/local-rag-chatbot.git
cd local-rag-chatbot
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env         # macOS/Linux: cp .env.example .env
```

Put your key(s) in `.env`, then start the server from the project folder:

```bash
python -m uvicorn app.main:app --port 8000
```

Open http://127.0.0.1:8000.

Start it with `python -m uvicorn` as shown. Running `python app/main.py` directly fails with a relative-import error.

All settings live in `.env`. The defaults are fine to start with; `.env.example` lists every option.

## How the failover works

The first version gave up the moment a model returned an error, and Gemini's free tier returns "503: high demand" a lot. Now `chat_with_fallback()` in `app/llm_provider.py` does this:

1. Call the model you picked.
2. If the error looks temporary (503, 429, "overloaded", timeout), wait 1.5 s and try once more.
3. If it still fails, try up to two other models, starting with a *different provider*. When one Gemini model is overloaded, the others usually are too, so falling back to another Gemini model rarely helps.

## Testing it

`eval_chatbot.py` checks retrieval quality, response time for every model, failover, memory and an end-to-end RAG answer. It only uses the standard library. Start the server, then run this in a second terminal:

```bash
python eval_chatbot.py              # everything
python eval_chatbot.py --skip-llm   # retrieval only, no API calls
```

My last run used 12 hand-written questions against a Python interview question bank (13 chunks). The document isn't in this repo, so to run the retrieval part yourself, upload your own file and edit `RETRIEVAL_SET` at the top of the script.

| | Result |
|---|---|
| Right chunk in top 5 / at #1 | 12 of 12 / 11 of 12 (MRR 0.958) |
| Search time | 15 ms median, 27 ms p95 |
| Off-topic questions ("capital of France?") | score 0.0, no context sent |
| Failed requests | 0 (two Gemini calls failed and were answered by Groq instead) |
| Full RAG answer | 2.2 s median, 4.2 s max |

These are small samples (one call per model, three RAG answers) on free tiers, and the timings change a lot from run to run. In one run the same Gemini model took 2 s; in the next it took 48 s.

## Project layout

```
app/
  main.py           API routes
  llm_provider.py   provider clients, thinking mode, retry and failover
  rag_store.py      chunking, TF-IDF / OpenAI embeddings, search
  memory_store.py   chat history per session
  schemas.py        request and response models
  config.py         settings from .env
static/index.html   the whole UI in one file, no build step
eval_chatbot.py     evaluation script
```

## Known limitations

- Chat history is kept in memory, so it's lost when the server restarts.
- TF-IDF matches words, not meaning. "Iterate" won't match "loop". Set `EMBEDDING_BACKEND=openai` if you need semantic search.
- There's no login. Keep it bound to `127.0.0.1` and don't expose it on a public network.

## Next up

- Save chat history to SQLite
- A bigger evaluation set (30+ questions) so the latency numbers mean something
- Stream replies token by token instead of waiting for the full answer

## License

MIT. See [LICENSE](LICENSE).
