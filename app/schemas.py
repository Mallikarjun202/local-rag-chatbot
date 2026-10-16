from pydantic import BaseModel, Field
from typing import List, Optional


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Any string identifying the conversation/user")
    message: str
    model: Optional[str] = None
    use_memory: Optional[bool] = Field(
        None, description="If omitted, uses the session's last saved memory setting (default: on)"
    )
    use_rag: bool = False
    thinking: bool = Field(False, description="Deeper reasoning: slower, better on hard questions")
    collection: Optional[str] = "default"

class ChatResponse(BaseModel):
    reply: str
    model: str
    used_memory: bool
    used_rag: bool
    sources: List[str] = []
    latency_ms: Optional[int] = None
    fallback_note: Optional[str] = None
    thinking: bool = False
    reasoning: Optional[str] = None

class MemoryToggleRequest(BaseModel):
    enabled: bool

class HistoryMessage(BaseModel):
    role: str
    content: str

class AddTextRequest(BaseModel):
    collection: str = "default"
    text: str
    source: str = "manual-text"
