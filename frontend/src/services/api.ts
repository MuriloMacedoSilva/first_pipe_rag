import type { ChatResponse, ChatSource } from "../types/chat"

const API_URL = (
  import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000"
).replace(/\/+$/, "")

const CONNECTION_ERROR = "Não foi possível conectar ao backend."
const SERVICE_ERROR = "O serviço de IA está temporariamente indisponível."
const GENERIC_ERROR = "Ocorreu um erro ao processar sua pergunta."

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null
}

function isChatSource(value: unknown): value is ChatSource {
  if (!isRecord(value)) {
    return false
  }

  return (
    typeof value.filename === "string" &&
    typeof value.relative_path === "string" &&
    (typeof value.section === "string" || value.section === null) &&
    typeof value.chunk_index === "number" &&
    (typeof value.distance === "number" || value.distance === null)
  )
}

function isChatResponse(value: unknown): value is ChatResponse {
  return (
    isRecord(value) &&
    typeof value.answer === "string" &&
    Array.isArray(value.sources) &&
    value.sources.every(isChatSource)
  )
}

async function readErrorDetail(response: Response): Promise<string | null> {
  try {
    const body: unknown = await response.json()
    if (isRecord(body) && typeof body.detail === "string") {
      return body.detail
    }
  } catch {
    return null
  }
  return null
}

export async function sendMessage(
  message: string,
  topK?: number,
): Promise<ChatResponse> {
  const payload: { message: string; top_k?: number } = { message }
  if (topK !== undefined) {
    payload.top_k = topK
  }

  let response: Response
  try {
    response = await fetch(`${API_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
  } catch {
    throw new Error(CONNECTION_ERROR)
  }

  if (!response.ok) {
    const detail = await readErrorDetail(response)
    if (response.status === 503) {
      throw new Error(SERVICE_ERROR)
    }
    throw new Error(detail ?? GENERIC_ERROR)
  }

  try {
    const body: unknown = await response.json()
    if (!isChatResponse(body)) {
      throw new Error(GENERIC_ERROR)
    }
    return body
  } catch (error) {
    if (error instanceof Error && error.message === GENERIC_ERROR) {
      throw error
    }
    throw new Error(GENERIC_ERROR)
  }
}
