"""
SQLAlchemy ORM models for memory persistence.
"""

import json
from datetime import datetime
from uuid import uuid4

from sqlalchemy import JSON, Column, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class MemoryRecordDB(Base):
    """Database model for memory records."""

    __tablename__ = "memory_records"

    # Primary key
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))

    # Core fields
    content = Column(Text, nullable=False, index=True)
    memory_type = Column(String(20), nullable=False, index=True, default="semantic")
    source = Column(String(255), nullable=True, index=True)
    confidence = Column(Float, nullable=True)

    # Metadata (stored as JSON)
    metadata = Column(JSON, nullable=True, default=dict)
    tags = Column(JSON, nullable=True, default=list)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Indexes for common queries
    __table_args__ = (
        Index("idx_memory_type_created", "memory_type", "created_at"),
        Index("idx_source_created", "source", "created_at"),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type,
            "source": self.source,
            "confidence": self.confidence,
            "metadata": self.metadata or {},
            "tags": self.tags or [],
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"<MemoryRecord(id={self.id}, type={self.memory_type}, content={self.content[:50]}...)>"
