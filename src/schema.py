from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

Priority = Literal["Low", "Med", "High"]

class Meta(BaseModel):
    total_turns: int

class Evidence(BaseModel):
    start_turn: int
    end_turn: int
    snippet: str

class ActionItem(BaseModel):
    owner: Optional[str] = None
    task: str
    due: Optional[str] = None
    priority: Priority = "Med"
    evidence: Evidence

class Decision(BaseModel):
    decision: str
    evidence: Evidence

class TopicSection(BaseModel):
    title: str
    summary_bullets: List[str] = Field(default_factory=list)
    decisions: List[Decision] = Field(default_factory=list)
    action_items: List[ActionItem] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    evidence: Evidence

class MoM(BaseModel):
    meeting_title: str = "MoM"
    tldr: str
    topics: List[TopicSection]
    decisions: List[Decision] = Field(default_factory=list)
    action_items: List[ActionItem] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    meta: Meta