import React, { useEffect, useRef, useState } from 'react'
import { api } from '../api.js'
import Markdown from '../components/Markdown.jsx'

export default function Chat() {
  const [sessions, setSessions] = useState([])
  const [sid, setSid] = useState('s1')
  const [msgs, setMsgs] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottom = useRef(null)

  const load = async (id) => {
    setSid(id)
    try { const d = await api.getSession(id); setMsgs(d.messages || []) } catch { setMsgs([]) }
  }
  useEffect(() => {
    api.listSessions().then(setSessions).catch(() => {})
    load('s1')
  }, [])
  useEffect(() => { bottom.current?.scrollIntoView({ behavior: 'smooth' }) }, [msgs])

  const send = async () => {
    if (!input.trim() || loading) return
    const q = input; setInput(''); setLoading(true)
    setMsgs((m) => [...m, { role: 'user', content: q }, { role: 'assistant', content: '', sources: [], steps: [] }])
    try {
      for await (const ev of api.streamChat(sid, q)) {
        if (ev.step) {
          setMsgs((m) => { const c = [...m]; c[c.length-1] = { ...c[c.length-1], steps: [...(c[c.length-1].steps || []), ev.step] }; return c })
        }
        if (ev.token) {
          setMsgs((m) => { const c = [...m]; c[c.length-1] = { ...c[c.length-1], content: c[c.length-1].content + ev.token }; return c })
        }
        if (ev.sources) {
          setMsgs((m) => { const c = [...m]; c[c.length-1] = { ...c[c.length-1], sources: ev.sources }; return c })
        }
      }
    } catch (e) {
      setMsgs((m) => { const c = [...m]; c[c.length-1] = { ...c[c.length-1], content: '❌ 后端未启动，请先 uvicorn main:app --port 8000' }; return c })
    }
    setLoading(false)
  }

  return (
    <div className="h-full flex">
      <aside className="w-56 bg-stone-900 text-stone-200 p-3 flex flex-col gap-2">
        <button onClick={() => { const id = 's' + Date.now(); setSessions((s) => [{ id, title: '新会话' }, ...s]); load(id) }}
          className="bg-stone-700 hover:bg-stone-600 rounded-lg py-2 text-sm">＋ 新会话</button>
        {(sessions.length ? sessions : [{ id: 's1', title: '默认会话' }]).map((s) => (
          <button key={s.id} onClick={() => load(s.id)}
            className={`text-left text-sm px-3 py-2 rounded-lg truncate ${sid === s.id ? 'bg-stone-700' : 'hover:bg-stone-800'}`}>
            💬 {s.title || s.id}
          </button>
        ))}
      </aside>
      <main className="flex-1 flex flex-col min-w-0">
        <div className="flex-1 overflow-y-auto p-6 space-y-4 max-w-3xl w-full mx-auto">
          {msgs.length === 0 && <div className="text-center text-stone-400 mt-20">👋 先去 📚 知识库页上传文档，再回来提问吧</div>}
          {msgs.map((m, i) => (
            <div key={i} className={m.role === 'user' ? 'flex justify-end' : 'flex justify-start'}>
              <div className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm ${m.role === 'user' ? 'whitespace-pre-wrap bg-orange-500 text-white' : 'bg-white border shadow-sm'}`}>
                {m.steps?.length > 0 && m.role === 'assistant' && (
                  <details open className="mb-2 text-xs text-stone-500 bg-stone-50 rounded-lg px-2 py-1.5">
                    <summary className="cursor-pointer select-none">🧠 思考过程（{m.steps.length} 步）</summary>
                    <div className="mt-1 space-y-0.5">{m.steps.map((s, k) => <div key={k}>· {s}</div>)}</div>
                  </details>
                )}
                {m.role === 'assistant'
                  ? <Markdown text={m.content} streaming={loading && i === msgs.length - 1} />
                  : m.content}
                {m.sources?.length > 0 && (
                  <div className="mt-3 pt-2 border-t text-xs text-stone-500 space-y-1">
                    <b>引用来源：</b>
                    {m.sources.map((s, j) => (
                      <div key={j} className="bg-stone-100 rounded px-2 py-1">📄 {s.source} · 第 {s.chunk + 1} 段 —— {s.snippet}…</div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
          <div ref={bottom} />
        </div>
        <div className="p-4 bg-white border-t">
          <div className="max-w-3xl mx-auto flex gap-2">
            <input value={input} onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && send()}
              placeholder="问点什么…（如：报销上限是多少？）"
              className="flex-1 border rounded-xl px-4 py-2.5 text-sm outline-none focus:ring-2 ring-orange-300" />
            <button onClick={send} disabled={loading} className="bg-orange-500 text-white px-5 rounded-xl text-sm disabled:opacity-50">
              {loading ? '思考中…' : '发送'}
            </button>
          </div>
        </div>
      </main>
    </div>
  )
}
