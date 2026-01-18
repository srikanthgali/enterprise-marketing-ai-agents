# Memory Management System

Comprehensive memory management for marketing agents with short-term, long-term, and session-scoped storage.

## Features

- **Short-term Memory**: Session-scoped storage with TTL support
  - In-memory backend (default)
  - Redis backend (optional for distributed systems)
- **Long-term Memory**: Persistent storage for historical data
  - JSON file storage
  - Vector store for semantic search
- **Session Management**: Workflow-scoped memory isolation
  - Automatic cleanup on session end
  - Conversation history tracking
  - Agent activity monitoring
- **Execution Records**: Audit trail for agent actions
- **Semantic Search**: Find similar memories using vector embeddings

## Quick Start

### Basic Usage

```python
from src.marketing_agents.memory import MemoryManager

# Initialize memory manager
memory_manager = MemoryManager()

# Save short-term memory
memory_manager.save(
    agent_id="marketing_strategy_agent",
    key="current_campaign",
    value={"name": "Q1 Launch", "budget": 50000},
    memory_type="short_term",
)

# Retrieve short-term memory
campaign = memory_manager.retrieve(
    agent_id="marketing_strategy_agent",
    key="current_campaign",
    memory_type="short_term",
)

# Save long-term memory
memory_manager.save(
    agent_id="marketing_strategy_agent",
    key="best_practices",
    value=["Personalize emails", "A/B test headlines"],
    memory_type="long_term",
)
```

### Session Context

```python
from src.marketing_agents.memory import create_session, MemoryManager

memory_manager = MemoryManager()

# Use context manager for automatic session management
with create_session(
    workflow_id="campaign_001",
    memory_manager=memory_manager,
    metadata={"user": "john@example.com"},
) as session:
    # Save session-scoped data
    session.save_agent_memory(
        agent_id="content_agent",
        key="draft",
        value="Campaign headline draft",
    )

    # Add conversation message
    session.add_conversation_message(
        agent_id="content_agent",
        role="assistant",
        content="Draft headline created",
    )

    # Get conversation history
    history = session.get_conversation_history()

# Session automatically cleaned up on exit
```

## Architecture

### Storage Backends

#### InMemoryBackend
- Default backend for short-term memory
- Python dictionary with TTL support
- Fast, no external dependencies
- Not persistent across restarts

#### RedisBackend
- Optional backend for short-term memory
- Distributed, persistent storage
- Automatic expiration with TTL
- Shared across multiple processes

### Memory Types

#### Short-term Memory
- **Purpose**: Temporary data for current task/session
- **Storage**: Redis or in-memory
- **TTL**: Configurable (default: 24 hours)
- **Use cases**:
  - Current workflow state
  - Temporary cache
  - Session variables

#### Long-term Memory
- **Purpose**: Historical data, learnings, persistent knowledge
- **Storage**: JSON files + FAISS vector store
- **TTL**: No expiration
- **Use cases**:
  - Historical executions
  - Learned patterns
  - Best practices
  - Campaign results

#### Shared Memory
- **Purpose**: Global knowledge base accessible to all agents
- **Storage**: FAISS indices in `data/embeddings/`
- **Access**: Read-only for agents
- **Use cases**:
  - Product documentation
  - Marketing guidelines
  - Industry knowledge

## API Reference

### MemoryManager

#### `save(agent_id, key, value, memory_type="short_term", ttl=None)`
Save data to agent's memory.

**Parameters:**
- `agent_id` (str): Agent identifier
- `key` (str): Memory key
- `value` (Any): Value to store (must be JSON serializable)
- `memory_type` (str): "short_term" or "long_term"
- `ttl` (int, optional): Time-to-live in seconds (short_term only)

**Returns:** `bool` - True if successful

#### `retrieve(agent_id, key, memory_type="short_term")`
Retrieve data from agent's memory.

**Parameters:**
- `agent_id` (str): Agent identifier
- `key` (str): Memory key
- `memory_type` (str): "short_term" or "long_term"

**Returns:** `Any` - Stored value or None

#### `search_similar(agent_id, query, top_k=5, filter_metadata=None)`
Search for similar memories using semantic search.

**Parameters:**
- `agent_id` (str): Agent identifier
- `query` (str): Search query
- `top_k` (int): Number of results to return
- `filter_metadata` (dict, optional): Additional metadata filters

**Returns:** `list` - List of matching memory entries with scores

#### `get_conversation_history(workflow_id, limit=None)`
Get conversation history for a workflow.

**Parameters:**
- `workflow_id` (str): Workflow identifier
- `limit` (int, optional): Maximum number of messages

**Returns:** `list` - List of conversation messages

#### `save_execution_record(agent_id, record)`
Save an execution record for audit trail.

**Parameters:**
- `agent_id` (str): Agent identifier
- `record` (dict): Execution details (input, output, status, etc.)

**Returns:** `str` - Record ID

#### `clear_session(workflow_id)`
Clear all session data for a workflow.

**Parameters:**
- `workflow_id` (str): Workflow identifier

**Returns:** `bool` - True if successful

### SessionContext

#### `save_agent_memory(agent_id, key, value, memory_type="short_term")`
Save data within session scope (automatically adds workflow_id prefix).

#### `retrieve_agent_memory(agent_id, key, memory_type="short_term")`
Retrieve data within session scope.

#### `add_conversation_message(agent_id, role, content, metadata=None)`
Add a message to conversation history.

#### `get_conversation_history(limit=None)`
Get conversation history for this session.

#### `save_execution_record(agent_id, record)`
Save execution record with workflow_id.

#### `get_session_summary()`
Get summary of session activity.

#### `end_session()`
End the session and perform cleanup.

## Configuration

Add to `.env`:

```env
# Memory storage paths
MEMORY_STORAGE_DIR=data/processed/memory
MEMORY_EXECUTION_RECORDS_DIR=data/processed/memory/execution_records

# Short-term memory
MEMORY_USE_REDIS=false
MEMORY_SHORT_TERM_TTL=86400

# Long-term memory
MEMORY_ENABLE_VECTOR_SEARCH=true
MEMORY_VECTOR_STORE=agent_memory

# Session settings
MEMORY_SESSION_TIMEOUT=3600
MEMORY_AUTO_CLEANUP=true
MEMORY_MAX_CONVERSATION_HISTORY=100

# Performance
MEMORY_CACHE_ENABLED=true
MEMORY_MAX_CACHE_SIZE=1000
```

Or use `config/settings.py`:

```python
from config.settings import get_settings

settings = get_settings()
memory_settings = settings.memory
```

## Integration with BaseAgent

The memory manager integrates seamlessly with `BaseAgent`:

```python
from src.marketing_agents.core.base_agent import BaseAgent
from src.marketing_agents.memory import MemoryManager

# Create memory manager (shared across all agents)
memory_manager = MemoryManager()

# Initialize agent with memory manager
agent = MarketingStrategyAgent(
    agent_id="strategy_001",
    config=config,
    memory_manager=memory_manager,
)

# Use in agent methods
agent.save_to_memory("key", "value", memory_type="short_term")
value = agent.retrieve_from_memory("key", memory_type="short_term")
```

## Examples

See `examples/memory_management_example.py` for comprehensive examples:

1. **Basic Operations**: Save and retrieve memory
2. **Session Context**: Workflow-scoped memory with auto-cleanup
3. **Execution Records**: Track agent execution history
4. **Semantic Search**: Find similar memories using vector search
5. **Multi-Agent Workflow**: Multiple agents sharing memory in a workflow

Run examples:

```bash
python examples/memory_management_example.py
```

## Storage Structure

```
data/
├── processed/
│   └── memory/
│       ├── long_term_memory.json          # Long-term memory JSON
│       └── execution_records/
│           ├── strategy_agent.jsonl       # Agent execution logs
│           ├── content_agent.jsonl
│           └── analytics_agent.jsonl
└── embeddings/
    └── agent_memory/                      # FAISS vector store
        ├── index.faiss
        └── index.pkl
```

## Best Practices

### 1. Use Appropriate Memory Types

- **Short-term**: Temporary, session-specific data
  ```python
  memory_manager.save(agent_id, "current_task", data, "short_term")
  ```

- **Long-term**: Historical data, learnings
  ```python
  memory_manager.save(agent_id, "campaign_results", data, "long_term")
  ```

### 2. Always Use Sessions for Workflows

```python
with create_session(workflow_id, memory_manager) as session:
    # All operations are isolated to this workflow
    session.save_agent_memory(agent_id, "data", value)
```

### 3. Track Important Executions

```python
record_id = memory_manager.save_execution_record(
    agent_id="analytics_agent",
    record={
        "status": "success",
        "input": input_data,
        "output": results,
        "duration_ms": duration,
    },
)
```

### 4. Use Semantic Search for Insights

```python
# Find relevant past experiences
similar = memory_manager.search_similar(
    agent_id="strategy_agent",
    query="successful email campaigns",
    top_k=5,
)
```

### 5. Monitor Memory Usage

```python
stats = memory_manager.get_stats()
print(f"Short-term entries: {stats['short_term_entries']}")
print(f"Long-term entries: {stats['long_term_entries']}")
print(f"Active sessions: {stats['active_sessions']}")
```

## Performance Considerations

1. **Redis for Production**: Use Redis backend for distributed systems
2. **TTL Management**: Set appropriate TTLs to avoid memory bloat
3. **Cleanup**: Enable auto-cleanup for sessions
4. **Vector Store**: Lazy-loaded, only initialized when semantic search is used
5. **Batch Operations**: Save multiple items before triggering vector store updates

## Testing

```bash
# Run memory management tests
pytest tests/unit/memory/test_memory_manager.py
pytest tests/integration/memory/test_memory_integration.py
```

## Troubleshooting

### Memory Not Persisting
- Check storage directory permissions
- Verify `MEMORY_STORAGE_DIR` path
- Ensure `_save_long_term_memory()` completes successfully

### Semantic Search Not Working
- Verify FAISS vector store exists
- Check `OPENAI_API_KEY` environment variable
- Ensure vector store path is correct

### Redis Connection Issues
- Verify Redis is running: `redis-cli ping`
- Check Redis connection settings in `.env`
- Test connection: `redis-cli -h <host> -p <port>`

### Session Cleanup Not Working
- Check `MEMORY_AUTO_CLEANUP=true` in config
- Verify context manager is being used (`with create_session...`)
- Ensure `end_session()` is called if not using context manager

## Future Enhancements

- [ ] Distributed session management with Redis
- [ ] Memory compression for large datasets
- [ ] Automatic memory archival
- [ ] Memory analytics dashboard
- [ ] Multi-tenancy support
- [ ] Memory replication across regions
