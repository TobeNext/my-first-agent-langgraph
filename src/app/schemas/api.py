from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class StreamMemory(BaseModel):
    thread: str = Field(min_length=1)
    resource: str = Field(min_length=1)


class MastraStreamRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    memory: StreamMemory
    maxSteps: int | None = None

    @property
    def thread_id(self) -> str:
        return self.memory.thread

    @property
    def resource_id(self) -> str:
        return self.memory.resource

    def last_user_message(self) -> str:
        for message in reversed(self.messages):
            if message.role == "user":
                return message.content
        return ""
