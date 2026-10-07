import type { FormEvent, KeyboardEvent } from "react"

interface ChatInputProps {
  value: string
  isLoading: boolean
  onChange: (value: string) => void
  onSubmit: () => void
}

export function ChatInput({
  value,
  isLoading,
  onChange,
  onSubmit,
}: ChatInputProps) {
  const canSubmit = value.trim().length > 0 && !isLoading

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (canSubmit) {
      onSubmit()
    }
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault()
      if (canSubmit) {
        onSubmit()
      }
    }
  }

  return (
    <form className="mx-auto w-full max-w-4xl" onSubmit={handleSubmit}>
      <label className="sr-only" htmlFor="chat-message">
        Pergunta sobre seus materiais
      </label>
      <div className="flex items-end gap-2 rounded-2xl border border-slate-300 bg-white p-2 shadow-[0_8px_30px_rgba(15,23,42,0.08)] transition focus-within:border-teal-700 focus-within:ring-3 focus-within:ring-teal-700/10 sm:gap-3 sm:p-3">
        <textarea
          aria-describedby="chat-hint"
          className="max-h-40 min-h-12 flex-1 resize-none bg-transparent px-2 py-3 text-base leading-6 text-slate-900 outline-none placeholder:text-slate-400 disabled:cursor-not-allowed disabled:opacity-60"
          disabled={isLoading}
          id="chat-message"
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Pergunte sobre seus materiais..."
          rows={1}
          value={value}
        />
        <button
          aria-label="Enviar pergunta"
          className="flex size-11 shrink-0 items-center justify-center rounded-xl bg-teal-800 text-white transition hover:bg-teal-700 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-700 disabled:cursor-not-allowed disabled:bg-slate-300"
          disabled={!canSubmit}
          type="submit"
        >
          <svg
            aria-hidden="true"
            className="size-5"
            fill="none"
            viewBox="0 0 24 24"
          >
            <path
              d="m5 12 14-7-4.5 14-3-5.5L5 12Z"
              stroke="currentColor"
              strokeLinejoin="round"
              strokeWidth="1.8"
            />
          </svg>
        </button>
      </div>
      <p className="mt-2 text-center text-[11px] text-slate-500" id="chat-hint">
        Enter envia · Shift + Enter cria uma nova linha
      </p>
    </form>
  )
}
