import type { ChatMessage } from "../types/chat"
import { Sources } from "./Sources"

interface MessageBubbleProps {
  message: ChatMessage
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user"
  const isError = message.status === "error"

  return (
    <article
      className={`flex items-end gap-2.5 ${isUser ? "justify-end" : "justify-start"}`}
    >
      {!isUser && (
        <div
          aria-hidden="true"
          className="mb-1 flex size-8 shrink-0 items-center justify-center rounded-full bg-teal-800 font-mono text-xs font-bold text-white"
        >
          R
        </div>
      )}

      <div
        className={`max-w-[86%] rounded-2xl px-4 py-3 text-[15px] leading-7 shadow-sm sm:max-w-[75%] sm:px-5 ${
          isUser
            ? "rounded-br-md bg-slate-900 text-slate-50"
            : isError
              ? "rounded-bl-md border border-red-200 bg-red-50 text-red-900"
              : "rounded-bl-md border border-slate-200 bg-white text-slate-800"
        }`}
      >
        <p className="whitespace-pre-wrap break-words">{message.content}</p>
        {!isUser && message.sources && (
          <Sources sources={message.sources} />
        )}
      </div>
    </article>
  )
}
