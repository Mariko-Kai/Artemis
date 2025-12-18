
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Text, Float, Boolean, JSON, LargeBinary
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
import uuid
from .database import Base

class MemoryRecordDB(Base):
    """
    SQLite-compatible MemoryRecord model for Artemis.
    Inherits from Artemis's shared declarative Base.
    """
    __tablename__ = "memory_records"

    # Primary key
    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))

    # Core fields
    content = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    memory_type = Column(String, nullable=False, default="semantic") # episodic, semantic
    status = Column(String, nullable=False, default="completed") # pending_summary, pending_embedding, completed
    source = Column(String, nullable=True) # e.g., "chat", "user_input"
    
    # Artemis Integration
    channel_id = Column(String, ForeignKey("sessions.id"), nullable=True) # Linked to ChatSession
    session_id = Column(String, nullable=True) # Redundant alias for channel_id if needed, but channel_id is broader

    confidence = Column(Float, nullable=True)
    importance = Column(Float, default=0.5)
    
    # Deduplication & Integrity
    content_hash = Column(String, nullable=True, index=True)
    
    # Versioning & Status
    version = Column(Integer, default=1)
    archived = Column(Boolean, default=False)

    # Metadata (SQLite has no native JSON, so we store as Text or rely on SQLAlchemy's JSON type to serialize)
    # SQLAlchemy's JSON type works on SQLite as Text with serializer
    metadata_json = Column("metadata", JSON, nullable=True)
    tags_json = Column("tags", JSON, nullable=True)
    entities_json = Column("entities", JSON, nullable=True)
    
    # Embeddings (Cached in SQLite to avoid re-computation if Model doesn't change)
    # Storing 1024 floats * 4 bytes = 4KB per record. 10k records = 40MB. Very feasible.
    # Stored as BLOB (bytes)
    embedding_blob = Column(LargeBinary, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None), onupdate=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    last_accessed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

    # Relationships
    # session = relationship("ChatSession", back_populates="memories") # Need to add back_populates to ChatSession if we want two-way

    def __repr__(self):
        return f"<MemoryRecordDB(id={self.id}, content={self.content[:30]}...)>"

class VectorMapping(Base):
    """
    Mapping between FAISS integer IDs and MemoryRecord UUIDs.
    """
    __tablename__ = "vector_mappings"

    faiss_id = Column(Integer, primary_key=True)
    memory_id = Column(String, ForeignKey("memory_records.id"), nullable=False)
    
    # Additional context for embedding
    embedding_model_version = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

    # Store full-dimension vector for re-ranking (MRL Stage 2)
    full_vector = Column(JSON, nullable=True)

class AuditLog(Base):
    """
    Simple audit log.
    """
    __tablename__ = "audit_log"
    
    id = Column(Integer, primary_key=True, index=True)
    memory_id = Column(String, index=True)
    operation = Column(String) # INSERT, QUERY, etc.
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    details = Column(String, nullable=True) # JSON string

class ConfigVersion(Base):
    """
    Config versioning.
    """
    __tablename__ = "config_versions"
    
    version_id = Column(Integer, primary_key=True)
    config_json = Column(JSON, nullable=False)
    effective_from = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

class JobQueue(Base):
    """
    Background job queue for summarization.
    """
    __tablename__ = "job_queue"
    
    id = Column(Integer, primary_key=True, index=True)
    job_type = Column(String, default="summarization")
    target_id = Column(String) # e.g., session_id
    status = Column(String, default="pending") # pending, processing, completed, failed
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    retries = Column(Integer, default=0)
    error = Column(Text, nullable=True)

class ArchivedMemoryRecordDB(Base):
    """
    Archived memory storage with compression.
    """
    __tablename__ = "archived_memory_records"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    original_id = Column(String, nullable=False, index=True) # Reference to original ID (not FK to allow deletion of original)
    archived_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None), index=True)
    
    # Content
    compressed_content = Column(LargeBinary, nullable=False) # Gzipped
    decompressed_size = Column(Integer, nullable=False)
    
    # Metadata preserved
    metadata_json = Column(JSON, nullable=True) # Includes embeddings, stats
    
    __table_args__ = ()
