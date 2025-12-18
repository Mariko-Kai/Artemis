"""
SQLAlchemy ORM models for memory persistence.
"""

import json
from datetime import datetime
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class MemoryRecordDB(Base):
    """
    Database model for memory records.
    Enhanced for Phase 3 with versioning, references, and archival.
    """

    __tablename__ = "memory_records"

    # Primary key
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))

    # Core fields
    content = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    memory_type = Column(String(20), nullable=False, default="semantic")
    status = Column(String(20), nullable=False, default="completed")
    source = Column(String(255), nullable=True)
    channel_id = Column(String(255), nullable=True) # Renamed from chat_id to generic channel/chat
    confidence = Column(Float, nullable=True)
    
    # Deduplication & Integrity
    content_hash = Column(String(64), nullable=True, index=True) # SHA256
    
    # Versioning for OCC
    version = Column(Integer, nullable=False, default=1)
    
    # Status
    archived = Column(Boolean, nullable=False, default=False)
    importance = Column(Float, nullable=True, default=0.0)

    # Metadata (stored as JSON)
    # 'metadata' is reserved by SQLAlchemy, so we use 'record_metadata' mapped to the column
    record_metadata = Column("metadata", JSON, nullable=True, default=dict)
    tags = Column(JSON, nullable=True, default=list)
    entities = Column(JSON, nullable=True, default=dict) # Extracted entities
    
    # References & Hierarchy
    ref_id = Column(JSON, nullable=True, default=list) # List of referenced IDs
    part_of_message_id = Column(String(36), nullable=True) # For chunking
    chunk_index = Column(Integer, nullable=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_accessed_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Indexes for common queries
    __table_args__ = (
        Index("idx_memory_type_created", "memory_type", "created_at"),
        Index("idx_source_created", "source", "created_at"),
        Index("idx_channel_created", "channel_id", "created_at"),
        # Partial index for archived=False is Postgres specific but useful
        Index("idx_archived_false_created", "created_at", postgresql_where=(archived == False)),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type,
            "source": self.source,
            "channel_id": self.channel_id,
            "confidence": self.confidence,
            "content_hash": self.content_hash,
            "version": self.version,
            "archived": self.archived,
            "importance": self.importance,
            "metadata": self.record_metadata or {},
            "tags": self.tags or [],
            "entities": self.entities or {},
            "ref_id": self.ref_id or [],
            "part_of_message_id": self.part_of_message_id,
            "chunk_index": self.chunk_index,
            "summary": self.summary,
            "status": self.status,
            "last_accessed_at": self.last_accessed_at.isoformat() if self.last_accessed_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"<MemoryRecord(id={self.id}, type={self.memory_type}, hash={self.content_hash})>"


class VectorMapping(Base):
    """
    Mapping between FAISS integer IDs and MemoryRecord UUIDs.
    Essential for persistent MRL/FAISS integration.
    """
    __tablename__ = "vector_mappings"

    faiss_id = Column(Integer, primary_key=True)
    memory_id = Column(String(36), ForeignKey("memory_records.id"), nullable=False)
    
    # Additional context for embedding
    embedding_model_version = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Store full-dimension vector for re-ranking (MRL Stage 2)
    full_vector = Column(JSON, nullable=True)
    
    __table_args__ = (
        Index("idx_vector_memory_id", "memory_id"),
    )

    def __repr__(self) -> str:
        return f"<VectorMapping(faiss_id={self.faiss_id}, memory_id={self.memory_id})>"


class AuditLog(Base):
    """
    Audit log for tracking operations on memories.
    """
    __tablename__ = "audit_log"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    memory_id = Column(String(36), index=True, nullable=True) # Nullable for bulk or config ops
    operation = Column(String(20), nullable=False) # INSERT, UPDATE, DELETE, QUERY
    user_id = Column(String(255), nullable=True) # User performing op
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    reason = Column(String(255), nullable=True)
    details = Column(JSON, nullable=True) # Diff or metadata

    def __repr__(self) -> str:
        return f"<AuditLog(op={self.operation}, user={self.user_id}, time={self.timestamp})>"


class ConfigVersion(Base):
    """
    Configuration version history.
    """
    __tablename__ = "config_versions"
    
    version_id = Column(Integer, primary_key=True, autoincrement=True)
    scope = Column(String(50), nullable=False, default="global") # global, org_id, chat_id
    config_json = Column(JSON, nullable=False)
    effective_from = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String(255), nullable=True)
    
    __table_args__ = (
        Index("idx_config_scope_effective", "scope", "effective_from"),
    )

    def __repr__(self) -> str:
        return f"<ConfigVersion(scope={self.scope}, v={self.version_id})>"
