import React from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom'
import Chat from './pages/Chat.jsx'
import KnowledgeBase from './pages/KnowledgeBase.jsx'
import './index.css'

function App() {
  return (
    <BrowserRouter>
      <div className="h-screen flex flex-col">
        <header className="flex items-center justify-between px-5 py-3 bg-white border-b">
          <b>🤖 AI 知识库问答台</b>
          <nav className="flex gap-4 text-sm">
            <Link to="/" className="text-orange-600">💬 对话</Link>
            <Link to="/kb" className="text-stone-600">📚 知识库管理</Link>
          </nav>
        </header>
        <div className="flex-1 min-h-0">
          <Routes>
            <Route path="/" element={<Chat />} />
            <Route path="/kb" element={<KnowledgeBase />} />
          </Routes>
        </div>
      </div>
    </BrowserRouter>
  )
}
createRoot(document.getElementById('root')).render(<App />)
