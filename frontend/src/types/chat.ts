export interface ChatSource {
  filename: string
  relative_path: string
  section: string | null
  chunk_index: number
  distance: number | null
}

export interface ChatResponse {
  answer: string
  sources: ChatSource[]
}

export interface ChatMessage {
  id: string
  role: "user" | "assistant"
  content: string
  sources?: ChatSource[]
  status?: "error"
}
