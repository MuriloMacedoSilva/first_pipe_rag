import type { ChatSource } from "../types/chat"

interface SourcesProps {
  sources: ChatSource[]
}

export function Sources({ sources }: SourcesProps) {
  if (sources.length === 0) {
    return null
  }

  return (
    <details className="mt-4 border-t border-slate-200 pt-3">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 text-xs font-semibold tracking-wide text-slate-600 uppercase focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-teal-700 [&::-webkit-details-marker]:hidden">
        <span>Fontes</span>
        <span className="rounded-full bg-slate-100 px-2 py-0.5 font-mono text-[11px] text-slate-500">
          {sources.length}
        </span>
      </summary>

      <ol className="mt-3 space-y-2">
        {sources.map((source, index) => (
          <li
            className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2.5"
            key={`${source.relative_path}-${source.chunk_index}-${index}`}
          >
            <p className="break-words font-mono text-xs leading-5 text-slate-700">
              {index + 1}. {source.relative_path}
            </p>
            {source.section && (
              <p className="mt-1 text-xs leading-5 text-slate-500">
                {source.section}
              </p>
            )}
          </li>
        ))}
      </ol>
    </details>
  )
}
