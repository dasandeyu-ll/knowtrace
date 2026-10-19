const API = '/api'

async function* streamChat(sessionId, message) {
  const res = await fetch(`${API}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, message }),
  })
  if (!res.ok || !res.body) throw new Error('请求失败')
  const reader = res.body.getReader()
  const dec = new TextDecoder()
  let buf = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buf += dec.decode(value, { stream: true })
    const lines = buf.split('\n')
    buf = lines.pop()
    for (const line of lines) {
      if (!line.startsWith('data:')) continue
      const raw = line.slice(5).trim()
      if (!raw) continue
      try { yield JSON.parse(raw) } catch {}
    }
  }
}

export const api = {
  listSessions: () => fetch(`${API}/sessions`).then(r => r.json()),
  getSession: (id) => fetch(`${API}/sessions/${id}`).then(r => r.json()),
  kbStatus: () => fetch(`${API}/kb/status`).then(r => r.json()),
  upload: (files) => {
    const fd = new FormData()
    for (const f of files) fd.append('files', f)
    return fetch(`${API}/kb/upload`, { method: 'POST', body: fd }).then(r => r.json())
  },
  streamChat,
}
