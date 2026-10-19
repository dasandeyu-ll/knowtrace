"""AI 知识库问答台后端：FastAPI + LangGraph ReAct + FAISS + SQLite"""
import io, json, os, sqlite3, time
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from pypdf import PdfReader

load_dotenv()
app = FastAPI(title="QA Agent")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DB = os.path.join(os.path.dirname(__file__), "data.db")
def db():
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row; return c
def init_db():
    c = db()
    c.execute("CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, title TEXT, created_at REAL)")
    c.execute("CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT, sources TEXT, created_at REAL)")
    c.commit(); c.close()
init_db()

# ---------- 知识库：混合检索（FAISS 向量 + 关键词），见 vector_store.py ----------
from vector_store import VectorStore, chunk_text, keyword_search, tokenize

CHUNKS_FILE = os.path.join(os.path.dirname(__file__), "chunks.json")
FAISS_FILE = os.path.join(os.path.dirname(__file__), "faiss.index")
store = VectorStore(FAISS_FILE, CHUNKS_FILE)

def autoload_seed():
    """索引为空时，自动导入项目根目录的示例文档，避免重启后手动重传"""
    if store.chunks: return
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isdir(root): return
    names = sorted(os.listdir(root))
    cands = [n for n in names if n.lower().endswith(".txt")]
    # 优先名字里带「测试/示例/test/sample」的文档，避免误把 README 当知识库
    keyed = [n for n in cands if any(k in n.lower() for k in ("测试", "示例", "test", "sample", "demo"))]
    for fn in (keyed or cands[:1]):
        try:
            with open(os.path.join(root, fn), encoding="utf-8", errors="ignore") as f:
                text = f.read()
            new = [{"text": ch, "source": fn, "chunk": idx} for idx, ch in enumerate(chunk_text(text))]
            store.add(new)
            print(f"[kb] 自动导入 {fn}，共 {len(new)} 块")
        except Exception as e:
            print(f"[kb] 自动导入失败 {fn}: {e}")
    store.build_index()
    store.save()

autoload_seed()

def retrieve(query: str, k: int = 4):
    return store.search(query, k=k)

class ChatReq(BaseModel):
    session_id: str = "default"
    message: str = ""

@app.get("/api/sessions")
def list_sessions():
    c = db(); rows = c.execute("SELECT * FROM sessions ORDER BY created_at DESC").fetchall(); c.close()
    return [dict(r) for r in rows]

@app.get("/api/sessions/{sid}")
def get_session(sid: str):
    c = db()
    rows = c.execute("SELECT role,content,sources FROM messages WHERE session_id=? ORDER BY id", (sid,)).fetchall()
    c.close()
    out = []
    for r in rows:
        out.append({"role": r["role"], "content": r["content"], "sources": json.loads(r["sources"] or "[]")})
    return {"session_id": sid, "messages": out}

@app.get("/api/kb/status")
def kb_status():
    return {"chunk_count": len(store.chunks),
            "doc_count": len(set(c["source"] for c in store.chunks)),
            "docs": sorted(set(c["source"] for c in store.chunks)),
            "retrieval_mode": store.mode,
            "embedding_model": os.getenv("EMBED_MODEL", "BAAI/bge-small-zh-v1.5") if store.index is not None else None,
            "vector_dim": store._dim or None}

@app.get("/api/kb/debug")
def kb_debug(q: str = "一线城市住宿上限"):
    """诊断：索引模式、向量维度、两路召回与融合结果"""
    vec = store._vector_search(q, k=4)
    kw = keyword_search(store.chunks, q, k=4)
    hits = retrieve(q)
    return {
        "retrieval_mode": store.mode,
        "vector_dim": store._dim or None,
        "chunk_count": len(store.chunks),
        "docs": sorted(set(c["source"] for c in store.chunks)),
        "query_tokens": tokenize(q),
        "vector_hits": [{"source": c["source"], "chunk": c["chunk"], "score": round(s, 4)} for c, s in vec],
        "keyword_hits": [{"source": c["source"], "chunk": c["chunk"], "score": round(s, 2)} for c, s in kw],
        "fused_hits": [{"source": h["source"], "chunk": h["chunk"], "snippet": h["snippet"]} for h in hits],
    }

@app.post("/api/kb/upload")
async def kb_upload(files: List[UploadFile] = File(...)):
    added = 0
    for f in files:
        raw = await f.read()
        name = f.filename or "doc.txt"
        if name.lower().endswith(".pdf"):
            try:
                pdf = PdfReader(io.BytesIO(raw))
                text = "\n".join([(p.extract_text() or "") for p in pdf.pages])
            except Exception: text = ""
        else:
            text = raw.decode("utf-8", errors="ignore")
        new = [{"text": ch, "source": name, "chunk": idx} for idx, ch in enumerate(chunk_text(text))]
        store.add(new)
        added += len(new)
    store.build_index()
    store.save()
    return {"ok": True, "added_chunks": added, "total": len(store.chunks), "retrieval_mode": store.mode}

def get_history(sid: str, limit: int = 12):
    """多轮记忆：取最近 N 条已落库消息，截断后喂进 Agent"""
    c = db()
    rows = c.execute("SELECT role,content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?", (sid, limit)).fetchall()
    c.close()
    hist = list(reversed([{"role": r["role"], "content": (r["content"] or "")[:1500]} for r in rows]))
    return hist

async def stream_agent_answer(query: str, history: list):
    """真流式：astream_events 边算边推。yield (kind, payload)；kind=step/token/sources"""
    messages = [("system", "你是企业知识库问答助手。必须先用knowledge_search检索，再基于检索结果回答；无相关内容就直说不知道。")]
    for h in history[-10:]:
        messages.append(("user" if h["role"] == "user" else "assistant", h["content"][:1500]))
    messages.append(("user", query))
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    from langchain_core.tools import tool
    @tool
    def knowledge_search(q: str) -> str:
        """当问题可能与上传文档有关时先调用此工具检索相关段落"""
        hs = retrieve(q)
        return json.dumps(hs, ensure_ascii=False) if hs else "知识库中无相关内容"
    llm = ChatOpenAI(
        model=os.getenv("CHAT_MODEL", "deepseek-chat"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
        temperature=0.7, streaming=True,
    )
    agent = create_react_agent(llm, [knowledge_search])
    state = {"messages": messages}
    full, tool_seen = "", False
    try:
        async for ev in agent.astream_events(state, version="v2"):
            kind = ev.get("event", "")
            if kind == "on_tool_start":
                tool_seen = True
                yield ("step", "🔍 Agent 决定调用 knowledge_search 检索…")
            elif kind == "on_tool_end":
                out = str(ev.get("data", {}).get("output", ""))[:120]
                n = out.count('"source"')
                yield ("step", f"📎 检索完成，约命中 {n} 段" if n else "📎 检索完成")
            elif kind == "on_chat_model_stream":
                tok = ""
                try: tok = ev["data"]["chunk"].content or ""
                except Exception: tok = ""
                if isinstance(tok, str) and tok:
                    full += tok
                    yield ("token", tok)
        if not tool_seen:
            yield ("step", "📎 本轮未触发检索（寒暄/通用问题）")
    except Exception as e:
        yield ("fallback", str(e))
    yield ("final", full)

def build_answer(query: str, sid: str = "default"):
    """无 Key 降级：本地模板回答（含多轮上下文感知提示）"""
    hits = retrieve(query)
    if not hits:
        return "知识库还没有相关文档（或还没上传）。请先到右上角 📚 知识库页上传 txt/md/pdf，我再基于文档回答。你也可以直接问通用问题。", []
    ctx = "\n".join([f"[{h['source']}第{h['chunk']+1}段]{h['text'][:400]}" for h in hits])
    return f"根据知识库找到 {len(hits)} 段相关内容，综合回答如下：\n\n{ctx[:1200]}\n\n（如需原文请看下方引用来源）", hits

@app.post("/api/chat")
async def chat(req: ChatReq):
    sid = req.session_id or "default"
    history = get_history(sid)
    c = db()
    if not c.execute("SELECT 1 FROM sessions WHERE id=?", (sid,)).fetchone():
        c.execute("INSERT INTO sessions VALUES(?,?,?)", (sid, req.message[:20] or "新会话", time.time()))
    c.execute("INSERT INTO messages(session_id,role,content,sources,created_at) VALUES(?,?,?,?,?)",
              (sid, "user", req.message, "[]", time.time())); c.commit(); c.close()

    use_llm = bool(os.getenv("OPENAI_API_KEY", "").startswith("sk-"))

    async def gen():
        import asyncio
        buf, srcs = "", []
        if use_llm:
            yield {"event": "message", "data": json.dumps({"step": f"🧠 理解问题（已载入 {len(history)} 条历史）…"}, ensure_ascii=False)}
            async for kind, payload in stream_agent_answer(req.message, history):
                if kind == "step":
                    yield {"event": "message", "data": json.dumps({"step": payload}, ensure_ascii=False)}
                elif kind == "token":
                    buf += payload
                    yield {"event": "message", "data": json.dumps({"token": payload}, ensure_ascii=False)}
                elif kind == "fallback":
                    yield {"event": "message", "data": json.dumps({"step": f"⚠️ Agent 异常，降级本地回答：{payload[:80]}"}, ensure_ascii=False)}
            hits = retrieve(req.message)
            srcs = [{"source": h["source"], "chunk": h["chunk"], "snippet": h["snippet"]} for h in hits]
            if not buf:
                buf, hits = build_answer(req.message, sid)
                srcs = [{"source": h["source"], "chunk": h["chunk"], "snippet": h["snippet"]} for h in hits]
                for ch in buf:
                    yield {"event": "message", "data": json.dumps({"token": ch}, ensure_ascii=False)}
                    await asyncio.sleep(0.008)
            else:
                names = ", ".join(f"{h['source']}第{h['chunk']+1}段" for h in hits[:4])
                yield {"event": "message", "data": json.dumps({"step": f"📚 引用来源已生成{('：' + names) if names else '（本轮无引用）'}"}, ensure_ascii=False)}
        else:
            yield {"event": "message", "data": json.dumps({"step": "🔍 本地检索模式（未配置 LLM Key）…"}, ensure_ascii=False)}
            await asyncio.sleep(0.2)
            buf, hits = build_answer(req.message, sid)
            srcs = [{"source": h["source"], "chunk": h["chunk"], "snippet": h["snippet"]} for h in hits]
            yield {"event": "message", "data": json.dumps({"step": f"📎 命中 {len(hits)} 段" if hits else "📎 无命中"}, ensure_ascii=False)}
            for ch in buf:
                yield {"event": "message", "data": json.dumps({"token": ch}, ensure_ascii=False)}
                await asyncio.sleep(0.01)
        yield {"event": "message", "data": json.dumps({"sources": srcs, "done": True}, ensure_ascii=False)}
        c2 = db()
        c2.execute("INSERT INTO messages(session_id,role,content,sources,created_at) VALUES(?,?,?,?,?)",
                   (sid, "assistant", buf, json.dumps(srcs, ensure_ascii=False), time.time())); c2.commit(); c2.close()
    return EventSourceResponse(gen())
