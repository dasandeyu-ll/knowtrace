import React from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

/**
 * 流式防闪烁：把「还没写完」的 Markdown 语法补全，
 * 否则刚收到 `**住宿` 而 `**` 还没闭合时会被当普通文本渲染，下一个 token 到了又突然变粗。
 * 注意：所有统计必须在原始文本上做，避免把补上去的符号重复计算。
 */
export function stabilize(text) {
  const src = text || ''
  let out = src
  const fences = (src.match(/```/g) || []).length
  const inline = src.replace(/```/g, '').match(/`/g)?.length || 0
  const bolds = (src.match(/\*\*/g) || []).length
  if (fences % 2 === 1) out += '\n```'
  if (inline % 2 === 1) out += '`'
  if (bolds % 2 === 1) out += '**'
  return out
}

const components = {
  p: ({ children }) => <p className="my-1.5 leading-relaxed first:mt-0 last:mb-0">{children}</p>,
  h1: ({ children }) => <h3 className="mt-3 mb-1.5 text-base font-bold text-stone-900">{children}</h3>,
  h2: ({ children }) => <h3 className="mt-3 mb-1.5 text-base font-bold text-stone-900">{children}</h3>,
  h3: ({ children }) => <h4 className="mt-3 mb-1 text-sm font-bold text-stone-900">{children}</h4>,
  h4: ({ children }) => <h4 className="mt-2 mb-1 text-sm font-semibold text-stone-900">{children}</h4>,
  ul: ({ children }) => <ul className="my-1.5 list-disc space-y-0.5 pl-5">{children}</ul>,
  ol: ({ children }) => <ol className="my-1.5 list-decimal space-y-0.5 pl-5">{children}</ol>,
  li: ({ children }) => <li className="leading-relaxed">{children}</li>,
  strong: ({ children }) => <strong className="font-semibold text-stone-900">{children}</strong>,
  em: ({ children }) => <em className="italic">{children}</em>,
  hr: () => <hr className="my-3 border-stone-200" />,
  blockquote: ({ children }) => (
    <blockquote className="my-2 border-l-2 border-orange-300 bg-stone-50 py-1 pl-3 text-stone-600">{children}</blockquote>
  ),
  a: ({ href, children }) => (
    <a href={href} target="_blank" rel="noreferrer" className="text-orange-600 underline decoration-dotted">
      {children}
    </a>
  ),
  table: ({ children }) => (
    <div className="my-2 overflow-x-auto">
      <table className="w-full border-collapse text-xs">{children}</table>
    </div>
  ),
  thead: ({ children }) => <thead className="bg-stone-100">{children}</thead>,
  th: ({ children }) => <th className="border border-stone-200 px-2 py-1 text-left font-semibold">{children}</th>,
  td: ({ children }) => <td className="border border-stone-200 px-2 py-1">{children}</td>,
  pre: ({ children }) => (
    <pre className="my-2 overflow-x-auto rounded-lg bg-stone-900 p-3 text-xs leading-relaxed text-stone-100">
      {children}
    </pre>
  ),
  code: ({ className, children }) => {
    const isBlock = Boolean(className) || String(children).includes('\n')
    if (isBlock) return <code className={className}>{children}</code>
    return (
      <code className="rounded border border-stone-200 bg-stone-100 px-1 py-0.5 font-mono text-[0.85em] text-rose-600">
        {children}
      </code>
    )
  },
}

export default function Markdown({ text, streaming }) {
  return (
    <div className="break-words">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {stabilize(text)}
      </ReactMarkdown>
      {streaming && <span className="ml-0.5 inline-block h-3.5 w-1.5 animate-pulse bg-stone-400 align-text-bottom" />}
    </div>
  )
}
