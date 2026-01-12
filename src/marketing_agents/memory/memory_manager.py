"""
Comprehensive Memory Management for Marketing Agents.

Provides:
- Short-term memory (in-memory dict or Redis)
- Long-term memory (vector store + JSON files)
- Session management (isolated per workflow_id)
- Conversation history tracking
- Execution record persistence
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from collections import defaultdict
import uuid

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from config.settings import get_settings

logger = logging.getLogger(__name__)


class MemoryBackend:
    """Base class for memory storage backends."""

    def get(self, key: str) -> Optional[Any]:
        """Retrieve value by key."""
        raise NotImplementedError

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store value with optional TTL."""
        raise NotImplementedError

    def delete(self, key: str) -> bool:
        """Delete value by key."""
        raise NotImplementedError

    def keys(self, pattern: str = "*") -> List[str]:
        """Get all keys matching pattern."""
        raise NotImplementedError

    def clear(self) -> bool:
        """Clear all data."""
        raise NotImplementedError


class InMemoryBackend(MemoryBackend):
    """In-memory dictionary backend with TTL support."""

    def __init__(self):
        self.data: Dict[str, Any] = {}
        self.expiry: Dict[str, float] = {}
        logger.info("InMemoryBackend initialized")

    def _cleanup_expired(self):
        """Remove expired keys."""
        now = time.time()
        expired_keys = [k for k, exp in self.expiry.items() if exp < now]
        for key in expired_keys:
            del self.data[key]
            del self.expiry[key]

    def get(self, key: str) -> Optional[Any]:
        """Retrieve value by key."""
        self._cleanup_expired()
        return self.data.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store value with optional TTL in seconds."""
        try:
            self.data[key] = value
            if ttl:
                self.expiry[key] = time.time() + ttl
            elif key in self.expiry:
                del self.expiry[key]
            return True
        except Exception as e:
            logger.error(f"Failed to set key {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete value by key."""
        try:
            if key in self.data:
                del self.data[key]
            if key in self.expiry:
                del self.expiry[key]
            return True
        except Exception as e:
            logger.error(f"Failed to delete key {key}: {e}")
            return False

    def keys(self, pattern: str = "*") -> List[str]:
        """Get all keys matching pattern (basic glob support)."""
        self._cleanup_expired()
        if pattern == "*":
            return list(self.data.keys())

        # Simple pattern matching
        import fnmatch

        return [k for k in self.data.keys() if fnmatch.fnmatch(k, pattern)]

    def clear(self) -> bool:
        """Clear all data."""
        try:
            self.data.clear()
            self.expiry.clear()
            return True
        except Exception as e:
            logger.error(f"Failed to clear data: {e}")
            return False


class RedisBackend(MemoryBackend):
    """Redis backend for distributed memory."""

    def __init__(self, redis_client):
        self.redis = redis_client
        logger.info("RedisBackend initialized")

    def get(self, key: str) -> Optional[Any]:
        """Retrieve value by key."""
        try:
            value = self.redis.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Redis get error for key {key}: {e}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store value with optional TTL in seconds."""
        try:
            serialized = json.dumps(value)
            if ttl:
                self.redis.setex(key, ttl, serialized)
            else:
                self.redis.set(key, serialized)
            return True
        except Exception as e:
            logger.error(f"Redis set error for key {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete value by key."""
        try:
            self.redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis delete error for key {key}: {e}")
            return False

    def keys(self, pattern: str = "*") -> List[str]:
        """Get all keys matching pattern."""
        try:
            keys = self.redis.keys(pattern)
            return [k.decode() if isinstance(k, bytes) else k for k in keys]
        except Exception as e:
            logger.error(f"Redis keys error: {e}")
            return []

    def clear(self) -> bool:
        """Clear all data (use with caution in production)."""
        try:
            self.redis.flushdb()
            return True
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return False


class MemoryManager:
    """
    Comprehensive memory management for marketing agents.

    Features:
    - Short-term memory: Session-scoped, in-memory or Redis
    - Long-term memory: Persistent, vector store + JSON
    - Shared memory: Global knowledge base (read-only)
    - Session management: Isolated per workflow_id
    - Conversation history: Track agent interactions
    - Execution records: Audit trail of agent actions
    """

    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        use_redis: bool = False,
        redis_client=None,
        embeddings_dir: Optional[Path] = None,
    ):
        """
        Initialize memory manager.

        Args:
            storage_dir: Directory for long-term memory JSON files
            use_redis: Use Redis for short-term memory
            redis_client: Redis client instance (if use_redis=True)
            embeddings_dir: Directory for vector stores
        """
        self.settings = get_settings()

        # Storage directories
        self.storage_dir = storage_dir or Path("data/processed/memory")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.embeddings_dir = embeddings_dir or Path(
            self.settings.vector_store.persist_directory
        )
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        # Short-term memory backend
        if use_redis and redis_client:
            self.short_term_backend = RedisBackend(redis_client)
            logger.info("Using Redis for short-term memory")
        else:
            self.short_term_backend = InMemoryBackend()
            logger.info("Using in-memory backend for short-term memory")

        # Long-term memory structures
        self.long_term_file = self.storage_dir / "long_term_memory.json"
        self.execution_records_dir = self.storage_dir / "execution_records"
        self.execution_records_dir.mkdir(exist_ok=True)

        # Session tracking
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.conversation_history: Dict[str, List[Dict]] = defaultdict(list)

        # Vector store for semantic search (lazy loaded)
        self._vector_store: Optional[FAISS] = None
        self._embeddings: Optional[OpenAIEmbeddings] = None

        # Load long-term memory from disk
        self._load_long_term_memory()

        logger.info(
            f"MemoryManager initialized (storage: {self.storage_dir}, "
            f"embeddings: {self.embeddings_dir})"
        )

    def _get_embeddings(self) -> OpenAIEmbeddings:
        """Get or create embeddings instance."""
        if not self._embeddings:
            self._embeddings = OpenAIEmbeddings(
                model=self.settings.vector_store.embedding_model
            )
        return self._embeddings

    def _get_vector_store(self) -> Optional[FAISS]:
        """Get or load vector store for semantic search."""
        if not self._vector_store:
            try:
                vector_store_path = self.embeddings_dir / "agent_memory"
                if vector_store_path.exists():
                    self._vector_store = FAISS.load_local(
                        str(vector_store_path),
                        self._get_embeddings(),
                        allow_dangerous_deserialization=True,
                    )
                    logger.info(f"Loaded vector store from {vector_store_path}")
                else:
                    # Create empty vector store
                    docs = [
                        Document(
                            page_content="Memory system initialized",
                            metadata={
                                "type": "system",
                                "timestamp": datetime.utcnow().isoformat(),
                            },
                        )
                    ]
                    self._vector_store = FAISS.from_documents(
                        docs, self._get_embeddings()
                    )
                    self._vector_store.save_local(str(vector_store_path))
                    logger.info(f"Created new vector store at {vector_store_path}")
            except Exception as e:
                logger.error(f"Failed to load/create vector store: {e}")
        return self._vector_store

    def _load_long_term_memory(self):
        """Load long-term memory from disk."""
        try:
            if self.long_term_file.exists():
                with open(self.long_term_file, "r") as f:
                    self.long_term_data = json.load(f)
                logger.info(
                    f"Loaded long-term memory ({len(self.long_term_data)} entries)"
                )
            else:
                self.long_term_data = {}
                logger.info("Initialized empty long-term memory")
        except Exception as e:
            logger.error(f"Failed to load long-term memory: {e}")
            self.long_term_data = {}

    def _save_long_term_memory(self):
        """Save long-term memory to disk."""
        try:
            with open(self.long_term_file, "w") as f:
                json.dump(self.long_term_data, f, indent=2, default=str)
            logger.debug(f"Saved long-term memory to {self.long_term_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save long-term memory: {e}")
            return False

    def _make_key(self, agent_id: str, key: str, memory_type: str) -> str:
        """Generate storage key."""
        return f"{memory_type}:{agent_id}:{key}"

    def save(
        self,
        agent_id: str,
        key: str,
        value: Any,
        memory_type: str = "short_term",
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Save data to agent's memory.

        Args:
            agent_id: Agent identifier
            key: Memory key
            value: Value to store (must be JSON serializable)
            memory_type: "short_term" or "long_term"
            ttl: Time-to-live in seconds (short_term only)

        Returns:
            True if successful
        """
        try:
            if memory_type == "short_term":
                storage_key = self._make_key(agent_id, key, memory_type)
                # Default TTL from settings if not specified
                ttl = ttl or self.settings.redis.ttl
                success = self.short_term_backend.set(storage_key, value, ttl)
                if success:
                    logger.debug(f"Saved short-term memory: {agent_id}/{key}")
                return success

            elif memory_type == "long_term":
                storage_key = self._make_key(agent_id, key, memory_type)
                self.long_term_data[storage_key] = {
                    "value": value,
                    "timestamp": datetime.utcnow().isoformat(),
                    "agent_id": agent_id,
                    "key": key,
                }
                success = self._save_long_term_memory()
                if success:
                    logger.debug(f"Saved long-term memory: {agent_id}/{key}")

                # Also add to vector store for semantic search
                try:
                    vector_store = self._get_vector_store()
                    if vector_store:
                        doc = Document(
                            page_content=str(value),
                            metadata={
                                "agent_id": agent_id,
                                "key": key,
                                "type": "long_term",
                                "timestamp": datetime.utcnow().isoformat(),
                            },
                        )
                        vector_store.add_documents([doc])
                        # Save updated vector store
                        vector_store.save_local(
                            str(self.embeddings_dir / "agent_memory")
                        )
                        logger.debug(f"Added to vector store: {agent_id}/{key}")
                except Exception as e:
                    logger.warning(f"Failed to add to vector store: {e}")

                return success

            else:
                logger.error(f"Invalid memory_type: {memory_type}")
                return False

        except Exception as e:
            logger.error(f"Failed to save memory {agent_id}/{key}: {e}")
            return False

    def retrieve(
        self,
        agent_id: str,
        key: str,
        memory_type: str = "short_term",
    ) -> Optional[Any]:
        """
        Retrieve data from agent's memory.

        Args:
            agent_id: Agent identifier
            key: Memory key
            memory_type: "short_term" or "long_term"

        Returns:
            Stored value or None
        """
        try:
            storage_key = self._make_key(agent_id, key, memory_type)

            if memory_type == "short_term":
                value = self.short_term_backend.get(storage_key)
                if value is not None:
                    logger.debug(f"Retrieved short-term memory: {agent_id}/{key}")
                return value

            elif memory_type == "long_term":
                entry = self.long_term_data.get(storage_key)
                if entry:
                    logger.debug(f"Retrieved long-term memory: {agent_id}/{key}")
                    return entry.get("value")
                return None

            else:
                logger.error(f"Invalid memory_type: {memory_type}")
                return None

        except Exception as e:
            logger.error(f"Failed to retrieve memory {agent_id}/{key}: {e}")
            return None

    def search_similar(
        self,
        agent_id: str,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar memories using semantic search.

        Args:
            agent_id: Agent identifier (filters results to this agent)
            query: Search query
            top_k: Number of results to return
            filter_metadata: Additional metadata filters

        Returns:
            List of matching memory entries with scores
        """
        try:
            vector_store = self._get_vector_store()
            if not vector_store:
                logger.warning("Vector store not available for semantic search")
                return []

            # Perform similarity search
            results = vector_store.similarity_search_with_score(query, k=top_k * 2)

            # Filter by agent_id and additional metadata
            filtered_results = []
            for doc, score in results:
                metadata = doc.metadata
                if metadata.get("agent_id") != agent_id:
                    continue

                # Apply additional filters
                if filter_metadata:
                    if not all(
                        metadata.get(k) == v for k, v in filter_metadata.items()
                    ):
                        continue

                filtered_results.append(
                    {
                        "content": doc.page_content,
                        "metadata": metadata,
                        "similarity_score": 1.0
                        / (1.0 + score),  # Convert distance to similarity
                        "key": metadata.get("key"),
                    }
                )

                if len(filtered_results) >= top_k:
                    break

            logger.debug(
                f"Found {len(filtered_results)} similar memories for {agent_id}"
            )
            return filtered_results

        except Exception as e:
            logger.error(f"Failed to search similar memories: {e}")
            return []

    def get_conversation_history(
        self,
        workflow_id: str,
        limit: Optional[int] = None,
    ) -> List[Dict]:
        """
        Get conversation history for a workflow.

        Args:
            workflow_id: Workflow identifier
            limit: Maximum number of messages to return (most recent first)

        Returns:
            List of conversation messages
        """
        history = self.conversation_history.get(workflow_id, [])
        if limit:
            history = history[-limit:]
        logger.debug(f"Retrieved {len(history)} messages for workflow {workflow_id}")
        return history

    def add_conversation_message(
        self,
        workflow_id: str,
        agent_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """
        Add a message to conversation history.

        Args:
            workflow_id: Workflow identifier
            agent_id: Agent identifier
            role: Message role ("user", "assistant", "system")
            content: Message content
            metadata: Additional metadata

        Returns:
            True if successful
        """
        try:
            message = {
                "timestamp": datetime.utcnow().isoformat(),
                "workflow_id": workflow_id,
                "agent_id": agent_id,
                "role": role,
                "content": content,
                "metadata": metadata or {},
            }
            self.conversation_history[workflow_id].append(message)
            logger.debug(f"Added conversation message to {workflow_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to add conversation message: {e}")
            return False

    def save_execution_record(
        self,
        agent_id: str,
        record: Dict[str, Any],
    ) -> str:
        """
        Save an execution record for audit trail.

        Args:
            agent_id: Agent identifier
            record: Execution details (input, output, status, etc.)

        Returns:
            Record ID
        """
        try:
            record_id = str(uuid.uuid4())
            record_with_meta = {
                "record_id": record_id,
                "agent_id": agent_id,
                "timestamp": datetime.utcnow().isoformat(),
                **record,
            }

            # Save to agent-specific file
            agent_records_file = self.execution_records_dir / f"{agent_id}.jsonl"
            with open(agent_records_file, "a") as f:
                f.write(json.dumps(record_with_meta, default=str) + "\n")

            logger.debug(f"Saved execution record {record_id} for {agent_id}")
            return record_id

        except Exception as e:
            logger.error(f"Failed to save execution record: {e}")
            return ""

    def get_execution_records(
        self,
        agent_id: str,
        limit: Optional[int] = None,
        status_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get execution records for an agent.

        Args:
            agent_id: Agent identifier
            limit: Maximum number of records to return
            status_filter: Filter by status ("success", "error", etc.)

        Returns:
            List of execution records
        """
        try:
            agent_records_file = self.execution_records_dir / f"{agent_id}.jsonl"
            if not agent_records_file.exists():
                return []

            records = []
            with open(agent_records_file, "r") as f:
                for line in f:
                    record = json.loads(line)
                    if status_filter and record.get("status") != status_filter:
                        continue
                    records.append(record)

            # Return most recent first
            records.reverse()
            if limit:
                records = records[:limit]

            logger.debug(f"Retrieved {len(records)} execution records for {agent_id}")
            return records

        except Exception as e:
            logger.error(f"Failed to get execution records: {e}")
            return []

    def clear_session(self, workflow_id: str) -> bool:
        """
        Clear all session data for a workflow.

        Args:
            workflow_id: Workflow identifier

        Returns:
            True if successful
        """
        try:
            # Clear conversation history
            if workflow_id in self.conversation_history:
                del self.conversation_history[workflow_id]

            # Clear session data
            if workflow_id in self.sessions:
                del self.sessions[workflow_id]

            # Clear short-term memory for this workflow
            pattern = f"short_term:*:workflow_{workflow_id}*"
            keys = self.short_term_backend.keys(pattern)
            for key in keys:
                self.short_term_backend.delete(key)

            logger.info(f"Cleared session data for workflow {workflow_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to clear session {workflow_id}: {e}")
            return False

    def create_session(self, workflow_id: str, metadata: Optional[Dict] = None) -> Dict:
        """
        Create a new session for a workflow.

        Args:
            workflow_id: Workflow identifier
            metadata: Additional session metadata

        Returns:
            Session information
        """
        session = {
            "workflow_id": workflow_id,
            "created_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }
        self.sessions[workflow_id] = session
        logger.info(f"Created session for workflow {workflow_id}")
        return session

    def get_session(self, workflow_id: str) -> Optional[Dict]:
        """Get session information for a workflow."""
        return self.sessions.get(workflow_id)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory manager statistics.

        Returns:
            Statistics dictionary
        """
        short_term_keys = self.short_term_backend.keys()
        return {
            "short_term_entries": len(short_term_keys),
            "long_term_entries": len(self.long_term_data),
            "active_sessions": len(self.sessions),
            "conversation_history_size": sum(
                len(msgs) for msgs in self.conversation_history.values()
            ),
            "storage_dir": str(self.storage_dir),
            "embeddings_dir": str(self.embeddings_dir),
            "vector_store_loaded": self._vector_store is not None,
        }
