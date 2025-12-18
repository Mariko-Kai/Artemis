"""
Metadata store implementation using SQLAlchemy.

Provides async database operations for memory persistence.
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ..interfaces import IMetadataStore
from ..models import MemoryRecord, MemoryStatus, MemoryType
from .models import Base, MemoryRecordDB

logger = logging.getLogger(__name__)


class MetadataStore(IMetadataStore):
    """Metadata persistence using SQLAlchemy with async support."""

    def __init__(self, database_url: str):
        """
        Initialize metadata store.

        Args:
            database_url: Async database connection URL
        """
        self.database_url = database_url
        self.engine = create_async_engine(database_url, echo=False)
        self.async_session = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
        logger.info(f"Metadata store initialized with {database_url}")

    async def init_db(self) -> None:
        """Initialize database tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created")

    def _to_db_model(self, record: MemoryRecord) -> MemoryRecordDB:
        """
        Convert Pydantic model to SQLAlchemy model.

        Args:
            record: Pydantic MemoryRecord

        Returns:
            SQLAlchemy MemoryRecordDB
        """
        return MemoryRecordDB(
            id=str(record.id),
            content=record.content,
            memory_type=record.memory_type.value,
            source=record.source,
            confidence=record.confidence,
            record_metadata=record.metadata,
            tags=record.tags,
            summary=record.summary,
            status=record.status.value,
            last_accessed_at=record.last_accessed_at,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )

    def _to_pydantic_model(self, db_record: MemoryRecordDB) -> MemoryRecord:
        """
        Convert SQLAlchemy model to Pydantic model.

        Args:
            db_record: SQLAlchemy MemoryRecordDB

        Returns:
            Pydantic MemoryRecord
        """
        return MemoryRecord(
            id=UUID(db_record.id),
            content=db_record.content,
            memory_type=MemoryType(db_record.memory_type),
            source=db_record.source,
            confidence=db_record.confidence,
            metadata=db_record.record_metadata or {},
            tags=db_record.tags or [],
            summary=db_record.summary,
            status=MemoryStatus(db_record.status),
            last_accessed_at=db_record.last_accessed_at,
            created_at=db_record.created_at,
            updated_at=db_record.updated_at,
        )

    async def save(self, record: MemoryRecord) -> MemoryRecord:
        """
        Save a memory record.

        Args:
            record: Memory record to save

        Returns:
            Saved memory record
        """
        async with self.async_session() as session:
            db_record = self._to_db_model(record)
            session.add(db_record)
            await session.commit()
            await session.refresh(db_record)
            logger.debug(f"Saved memory record {db_record.id}")
            return self._to_pydantic_model(db_record)

    async def batch_save(self, records: List[MemoryRecord]) -> List[MemoryRecord]:
        """
        Save multiple memory records.

        Args:
            records: List of memory records

        Returns:
            List of saved records
        """
        if not records:
            return []

        async with self.async_session() as session:
            db_records = [self._to_db_model(record) for record in records]
            session.add_all(db_records)
            await session.commit()
            logger.info(f"Batch saved {len(db_records)} memory records")

            # Convert back to Pydantic models
            return [self._to_pydantic_model(db_record) for db_record in db_records]

    async def get(self, id: str) -> Optional[MemoryRecord]:
        """
        Retrieve a memory record by ID.

        Args:
            id: Memory record ID

        Returns:
            Memory record if found, None otherwise
        """
        async with self.async_session() as session:
            result = await session.get(MemoryRecordDB, id)
            if result:
                return self._to_pydantic_model(result)
            return None

    async def query(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[MemoryRecord]:
        """
        Query memory records with filters.

        Args:
            filters: Filter criteria (e.g., {"memory_type": "semantic"})
            limit: Maximum results
            offset: Results to skip

        Returns:
            List of matching records
        """
        async with self.async_session() as session:
            stmt = select(MemoryRecordDB)

            # Apply filters
            if filters:
                if "memory_type" in filters:
                    stmt = stmt.where(MemoryRecordDB.memory_type == filters["memory_type"])
                if "source" in filters:
                    stmt = stmt.where(MemoryRecordDB.source == filters["source"])
                if "tags" in filters:
                    # For JSON array containment (SQLite doesn't support well, basic check)
                    # In production, consider PostgreSQL with proper JSON operators
                    pass

            # Apply pagination
            stmt = stmt.order_by(MemoryRecordDB.created_at.desc())
            stmt = stmt.limit(limit).offset(offset)

            # Execute query
            result = await session.execute(stmt)
            db_records = result.scalars().all()

            return [self._to_pydantic_model(record) for record in db_records]

    async def update(self, id: str, updates: Dict[str, Any]) -> Optional[MemoryRecord]:
        """
        Update a memory record.

        Args:
            id: Record ID
            updates: Fields to update

        Returns:
            Updated record if found, None otherwise
        """
        async with self.async_session() as session:
            db_record = await session.get(MemoryRecordDB, id)
            if not db_record:
                return None

            # Update fields
            for key, value in updates.items():
                if hasattr(db_record, key):
                    setattr(db_record, key, value)

            await session.commit()
            await session.refresh(db_record)
            logger.debug(f"Updated memory record {id}")
            return self._to_pydantic_model(db_record)

    async def delete(self, id: str) -> bool:
        """
        Delete a memory record.

        Args:
            id: Record ID

        Returns:
            True if deleted, False if not found
        """
        async with self.async_session() as session:
            db_record = await session.get(MemoryRecordDB, id)
            if not db_record:
                return False

            await session.delete(db_record)
            await session.commit()
            logger.debug(f"Deleted memory record {id}")
            return True

    async def search(self, query: str, filters: Optional[Dict[str, Any]] = None, limit: int = 20) -> List[MemoryRecord]:
        """
        Search for memory records in storage using text matching.
        """
        async with self.async_session() as session:
            search_str = f"%{query}%"
            stmt = select(MemoryRecordDB).where(
                or_(
                    MemoryRecordDB.content.like(search_str),
                    MemoryRecordDB.summary.like(search_str)
                )
            ).where(MemoryRecordDB.status != "completed")

            if filters:
                if "memory_type" in filters:
                    stmt = stmt.where(MemoryRecordDB.memory_type == filters["memory_type"])
                if "source" in filters:
                    stmt = stmt.where(MemoryRecordDB.source == filters["source"])

            stmt = stmt.order_by(MemoryRecordDB.created_at.desc()).limit(limit)
            result = await session.execute(stmt)
            return [self._to_pydantic_model(r) for r in result.scalars().all()]

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with statistics
        """
        async with self.async_session() as session:
            # Total count
            count_stmt = select(MemoryRecordDB)
            result = await session.execute(count_stmt)
            total = len(result.scalars().all())

            # By type
            type_counts = {}
            for memory_type in MemoryType:
                type_stmt = select(MemoryRecordDB).where(
                    MemoryRecordDB.memory_type == memory_type.value
                )
                type_result = await session.execute(type_stmt)
                type_counts[memory_type.value] = len(type_result.scalars().all())

            return {
                "total_memories": total,
                "memory_types_count": type_counts,
            }
