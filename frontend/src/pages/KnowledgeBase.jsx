import React, { useEffect, useState } from 'react'
import { api } from '../api.js'

export default function KnowledgeBase() {
  const [status, setStatus] = useState({ chunk_count: 0, doc_count: 0 })
  const [msg, setMsg] = useState('')
  const refresh = () => api.kbStatus().then(setStatus).catch(() => {})
  useEffect(() => { refresh() }, [])

  const onFiles = async (e) => {
    const files = [...e.target.files]
    if (!files.length) return
    setMsg('上传切块入库中…')
    try {
      const r = await api.upload(files)
      setMsg(`✅ 入库成功：新增 ${r.added_chunks} 块，当前共 ${r.total} 块`)
      refresh()
    } catch { setMsg('❌ 上传失败，确认后端已启动') }
  }

  return (
    <div className="max-w-2xl mx-auto p-8">
      <h2 className="text-xl font-bold mb-2">📚 知识库管理</h2>
      <p className="text-sm text-stone-500 mb-6">上传 txt / md / pdf → 自动切块（800字/150重叠）→ 向量入库 → 对话页可引用</p>
      <div className="bg-white border rounded-2xl p-6 text-center">
        <label className="inline-block bg-stone-900 text-white text-sm px-6 py-2.5 rounded-xl cursor-pointer hover:bg-stone-700">
          选择文件上传
          <input type="file" multiple accept=".txt,.md,.pdf" className="hidden" onChange={onFiles} />
        </label>
        <div className="mt-4 text-sm text-stone-600">📄 文档数 {status.doc_count} · 🧩 切块数 {status.chunk_count}</div>
        {msg && <div className="mt-3 text-sm">{msg}</div>}
      </div>
    </div>
  )
}
