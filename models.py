"""SQLAlchemy models for parent-child document chunking."""
from __future__ import annotations

import uuid

from sqlalchemy import Column, String, Text, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class ParentDocument(Base):
    """Parent table: stores macro-level long text segments or chapter summaries."""
    __tablename__ = 'parent_documents'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    doc_id = Column(String(36), nullable=False, index=True)
    file_name = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    full_parent_text = Column(Text, nullable=False)
    security_level = Column(String(20), default="public")
    chunk_index = Column(String(10), default="0")

    children = relationship("ChildChunk", back_populates="parent", cascade="all, delete-orphan")


class ChildChunk(Base):
    """Child table: stores fine-grained text segments mapped to vector store."""
    __tablename__ = 'child_chunks'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    parent_id = Column(String(36), ForeignKey('parent_documents.id', ondelete='CASCADE'), nullable=True)
    doc_id = Column(String(36), nullable=False, index=True)
    child_text = Column(Text, nullable=False)
    qdrant_point_id = Column(String(36))
    chunk_index = Column(String(10), default="0")

    parent = relationship("ParentDocument", back_populates="children")
