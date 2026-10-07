import { useEffect, useRef, useState } from "react"

import { sendMessage } from "../services/api"
import type { ChatMessage } from "../types/chat"
import { ChatInput } from "./ChatInput"
import { MessageBubble } from "./MessageBubble"

const SUGGESTIONS = [
  "O que é backpropagation?",
  "Explique a diferença entre Ridge e Lasso.",
  "Quais funções de ativação aparecem no material?",
]

function createMessageId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID()
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

export function Chat() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const inFlightRef = useRef(false)
  const endOfMessagesRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, isLoading])

  async function submitMessage(value = input) {
    const message = value.trim()
    if (!message || inFlightRef.current) {
      return
    }

    inFlightRef.current = true
    setMessages((current) => [
      ...current,
      { id: createMessageId(), role: "user", content: message },
    ])
    setInput("")
    setIsLoading(true)

    try {
      const response = await sendMessage(message)
      setMessages((current) => [
        ...current,
        {
          id: createMessageId(),
          role: "assistant",
          content: response.answer,
          sources: response.sources,
        },
      ])
    } catch (error) {
      const content =
        error instanceof Error
          ? error.message
          : "Não foi possível obter uma resposta. Tente novamente."
      setMessages((current) => [
        ...current,
        {
          id: createMessageId(),
          role: "assistant",
          content,
          status: "error",
        },
      ])
    } finally {
      inFlightRef.current = false
      setIsLoading(false)
    }
  }

  return (
    <div className="mx-auto flex h-dvh min-h-screen w-full max-w-[920px] flex-col border-slate-200 bg-[#fbfaf7] sm:border-x sm:shadow-[0_0_60px_rgba(15,23,42,0.06)]">
      <header className="shrink-0 border-b border-slate-200 bg-[#fbfaf7]/95 px-4 py-4 sm:px-8 sm:py-5">
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="flex size-10 items-center justify-center rounded-xl bg-teal-800 font-mono text-sm font-bold text-white shadow-sm">
              R
            </div>
            <div>
              <h1 className="font-display text-xl font-semibold tracking-tight text-slate-950">
                RAG Chat
              </h1>
              <p className="text-sm text-slate-500">
                Pergunte sobre seus materiais
              </p>
            </div>
          </div>
          <div className="hidden items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 sm:flex">
            <span className="size-1.5 rounded-full bg-emerald-500" />
            Documentação local
          </div>
        </div>
      </header>

      <main className="min-h-0 flex-1 overflow-y-auto px-4 py-6 sm:px-8 sm:py-8">
        {messages.length === 0 ? (
          <section className="mx-auto flex min-h-full max-w-2xl flex-col items-center justify-center py-8 text-center">
            <p className="mb-3 font-mono text-xs font-semibold tracking-[0.18em] text-teal-800 uppercase">
              Base de conhecimento
            </p>
            <h2 className="font-display text-3xl font-semibold tracking-tight text-slate-950 sm:text-4xl">
              O que você quer descobrir?
            </h2>
            <p className="mt-3 max-w-md text-base leading-7 text-slate-600">
              Faça uma pergunta sobre o material disponível e receba uma
              resposta acompanhada das fontes consultadas.
            </p>
            <div className="mt-8 grid w-full gap-2 sm:grid-cols-3">
              {SUGGESTIONS.map((suggestion) => (
                <button
                  className="rounded-xl border border-slate-200 bg-white px-4 py-3 text-left text-sm leading-5 text-slate-700 transition hover:border-teal-700 hover:text-teal-900 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-700 disabled:cursor-not-allowed disabled:opacity-50"
                  disabled={isLoading}
                  key={suggestion}
                  onClick={() => void submitMessage(suggestion)}
                  type="button"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </section>
        ) : (
          <div className="space-y-6">
            {messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))}

            {isLoading && (
              <div
                aria-live="polite"
                className="flex items-end gap-2.5"
                role="status"
              >
                <div className="mb-1 flex size-8 items-center justify-center rounded-full bg-teal-800 font-mono text-xs font-bold text-white">
                  R
                </div>
                <div className="rounded-2xl rounded-bl-md border border-slate-200 bg-white px-4 py-3 text-sm text-slate-500 shadow-sm">
                  <span>Pensando</span>
                  <span className="loading-dots" aria-hidden="true">
                    ...
                  </span>
                </div>
              </div>
            )}
          </div>
        )}
        <div ref={endOfMessagesRef} />
      </main>

      <footer className="shrink-0 border-t border-slate-200 bg-[#f6f4ee] px-3 py-3 sm:px-8 sm:py-5">
        <ChatInput
          isLoading={isLoading}
          onChange={setInput}
          onSubmit={() => void submitMessage()}
          value={input}
        />
      </footer>
    </div>
  )
}
