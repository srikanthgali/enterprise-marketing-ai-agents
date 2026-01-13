"""
Centralized Settings and Configuration Management.

Loads configuration from environment variables, YAML files, and provides
validated settings for the entire application.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic.types import SecretStr
import yaml
from functools import lru_cache


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    level: str = Field(default="INFO", env="LOG_LEVEL")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", env="LOG_FORMAT"
    )
    file_enabled: bool = Field(default=True, env="LOG_FILE_ENABLED")
    file_path: str = Field(default="logs/app.log", env="LOG_FILE_PATH")
    file_max_bytes: int = Field(default=10485760, env="LOG_FILE_MAX_BYTES")  # 10MB
    file_backup_count: int = Field(default=5, env="LOG_FILE_BACKUP_COUNT")
    console_enabled: bool = Field(default=True, env="LOG_CONSOLE_ENABLED")

    @field_validator("level")
    @classmethod
    def validate_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()


class APISettings(BaseSettings):
    """API configuration."""

    openai_api_key: SecretStr = Field(..., env="OPENAI_API_KEY")
    anthropic_api_key: Optional[SecretStr] = Field(None, env="ANTHROPIC_API_KEY")
    serper_api_key: Optional[SecretStr] = Field(None, env="SERPER_API_KEY")

    openai_org_id: Optional[str] = Field(None, env="OPENAI_ORG_ID")
    openai_timeout: int = Field(default=60, env="OPENAI_TIMEOUT")
    openai_max_retries: int = Field(default=3, env="OPENAI_MAX_RETRIES")

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="allow"
    )


class RedisSettings(BaseSettings):
    """Redis configuration."""

    enabled: bool = Field(default=True, env="REDIS_ENABLED")
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    db: int = Field(default=0, env="REDIS_DB")
    password: Optional[SecretStr] = Field(None, env="REDIS_PASSWORD")
    ttl: int = Field(default=86400, env="REDIS_TTL")  # 24 hours

    @property
    def url(self) -> str:
        """Get Redis URL."""
        password = f":{self.password.get_secret_value()}@" if self.password else ""
        return f"redis://{password}{self.host}:{self.port}/{self.db}"


class VectorStoreSettings(BaseSettings):
    """Vector store configuration."""

    type: str = Field(default="faiss", env="VECTOR_STORE_TYPE")
    dimension: int = Field(default=1536, env="VECTOR_DIMENSION")
    index_type: str = Field(default="IVFFlat", env="VECTOR_INDEX_TYPE")
    metric: str = Field(default="cosine", env="VECTOR_METRIC")
    persist_directory: str = Field(default="data/embeddings", env="VECTOR_PERSIST_DIR")

    # Embedding model configuration
    embedding_model: str = Field(
        default="text-embedding-3-small", env="EMBEDDING_MODEL"
    )
    embedding_batch_size: int = Field(default=100, env="EMBEDDING_BATCH_SIZE")
    embedding_cache_enabled: bool = Field(default=True, env="EMBEDDING_CACHE_ENABLED")


class SystemSettings(BaseSettings):
    """System-level configuration."""

    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    max_concurrent_agents: int = Field(default=5, env="MAX_CONCURRENT_AGENTS")
    handoff_timeout: int = Field(default=300, env="HANDOFF_TIMEOUT")
    max_retries: int = Field(default=3, env="MAX_RETRIES")

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v):
        """Validate environment."""
        valid_envs = ["development", "staging", "production"]
        if v.lower() not in valid_envs:
            raise ValueError(f"Environment must be one of {valid_envs}")
        return v.lower()


class MemorySettings(BaseSettings):
    """Memory management configuration."""

    # Storage paths
    storage_dir: str = Field(default="data/processed/memory", env="MEMORY_STORAGE_DIR")
    execution_records_dir: str = Field(
        default="data/processed/memory/execution_records",
        env="MEMORY_EXECUTION_RECORDS_DIR",
    )
    vector_store_dir: str = Field(
        default="data/embeddings", env="MEMORY_VECTOR_STORE_DIR"
    )
    vector_store_name: str = Field(
        default="agent_memory", env="MEMORY_VECTOR_STORE_NAME"
    )

    # Backend settings - Redis is default for production reliability
    backend_type: str = Field(
        default="redis", env="MEMORY_BACKEND_TYPE"
    )  # "redis" or "memory"
    use_redis: bool = Field(default=True, env="MEMORY_USE_REDIS")  # Changed to True

    # TTL settings
    short_term_ttl: int = Field(default=86400, env="MEMORY_SHORT_TERM_TTL")  # 24 hours
    session_timeout: int = Field(default=3600, env="MEMORY_SESSION_TIMEOUT")  # 1 hour
    search_cache_ttl: int = Field(
        default=1800, env="MEMORY_SEARCH_CACHE_TTL"
    )  # 30 minutes

    # Long-term memory settings
    enable_vector_search: bool = Field(default=True, env="MEMORY_ENABLE_VECTOR_SEARCH")
    embedding_model: str = Field(
        default="text-embedding-3-small", env="MEMORY_EMBEDDING_MODEL"
    )
    embedding_dimensions: int = Field(default=1536, env="MEMORY_EMBEDDING_DIMENSIONS")
    embedding_batch_size: int = Field(default=100, env="MEMORY_EMBEDDING_BATCH_SIZE")

    # Vector search settings
    vector_search_top_k: int = Field(default=5, env="MEMORY_VECTOR_SEARCH_TOP_K")
    similarity_threshold: float = Field(default=0.7, env="MEMORY_SIMILARITY_THRESHOLD")
    enable_reranking: bool = Field(default=True, env="MEMORY_ENABLE_RERANKING")

    # Session settings
    auto_cleanup: bool = Field(default=True, env="MEMORY_AUTO_CLEANUP")
    max_conversation_history: int = Field(
        default=100, env="MEMORY_MAX_CONVERSATION_HISTORY"
    )
    max_session_duration: int = Field(
        default=14400, env="MEMORY_MAX_SESSION_DURATION"
    )  # 4 hours
    persist_sessions: bool = Field(default=True, env="MEMORY_PERSIST_SESSIONS")

    # Performance settings
    cache_enabled: bool = Field(default=True, env="MEMORY_CACHE_ENABLED")
    max_cache_size: int = Field(default=1000, env="MEMORY_MAX_CACHE_SIZE")
    cache_policy: str = Field(
        default="lru", env="MEMORY_CACHE_POLICY"
    )  # "lru" or "lfu"
    lazy_load_vector_store: bool = Field(
        default=True, env="MEMORY_LAZY_LOAD_VECTOR_STORE"
    )

    # Execution records
    execution_records_enabled: bool = Field(
        default=True, env="MEMORY_EXECUTION_RECORDS_ENABLED"
    )
    execution_records_format: str = Field(
        default="jsonl", env="MEMORY_EXECUTION_RECORDS_FORMAT"
    )  # "jsonl" or "json"
    max_records_per_file: int = Field(default=10000, env="MEMORY_MAX_RECORDS_PER_FILE")
    retention_days: int = Field(default=90, env="MEMORY_RETENTION_DAYS")

    # Monitoring
    monitoring_enabled: bool = Field(default=True, env="MEMORY_MONITORING_ENABLED")
    stats_interval: int = Field(default=300, env="MEMORY_STATS_INTERVAL")  # 5 minutes
    alert_short_term_high: int = Field(
        default=10000, env="MEMORY_ALERT_SHORT_TERM_HIGH"
    )
    alert_active_sessions_high: int = Field(
        default=100, env="MEMORY_ALERT_ACTIVE_SESSIONS_HIGH"
    )


class SecuritySettings(BaseSettings):
    """Security configuration."""

    authentication_required: bool = Field(default=True, env="AUTHENTICATION_REQUIRED")
    api_key_rotation_days: int = Field(default=90, env="API_KEY_ROTATION_DAYS")
    max_requests_per_minute: int = Field(default=100, env="MAX_REQUESTS_PER_MINUTE")
    # Use Union[str, List[str]] to prevent automatic JSON parsing
    allowed_origins: Union[str, List[str]] = Field(
        default="localhost,127.0.0.1", env="ALLOWED_ORIGINS"
    )

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_allowed_origins(cls, v):
        """Parse comma-separated string into list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v


class FastAPISettings(BaseSettings):
    """FastAPI application settings."""

    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    reload: bool = Field(default=True, env="API_RELOAD")
    workers: int = Field(default=1, env="API_WORKERS")
    log_level: str = Field(default="info", env="API_LOG_LEVEL")

    # CORS settings
    cors_enabled: bool = Field(default=True, env="CORS_ENABLED")

    # Rate limiting
    rate_limit_enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_period: int = Field(default=60, env="RATE_LIMIT_PERIOD")  # seconds

    # Request timeouts
    request_timeout: int = Field(default=300, env="REQUEST_TIMEOUT")  # seconds

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid = ["debug", "info", "warning", "error", "critical"]
        if v.lower() not in valid:
            raise ValueError(f"Log level must be one of {valid}")
        return v.lower()


class Settings(BaseSettings):
    """Main application settings."""

    # Project metadata
    project_name: str = "Enterprise Marketing AI Agents"
    version: str = "0.1.0"
    description: str = "Multi-agent AI system for marketing automation"

    # Base paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    config_dir: Path = Field(default_factory=lambda: Path(__file__).parent)
    data_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "data"
    )
    logs_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "logs"
    )

    # Component settings
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    api: APISettings = Field(default_factory=APISettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    system: SystemSettings = Field(default_factory=SystemSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    fastapi: FastAPISettings = Field(default_factory=FastAPISettings)

    # Agent configuration
    agents_config: Optional[Dict[str, Any]] = None
    models_config: Optional[Dict[str, Any]] = None
    memory_config: Optional[Dict[str, Any]] = None
    api_config: Optional[Dict[str, Any]] = None

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="allow"
    )

    def __init__(self, **kwargs):
        """Initialize settings and load configuration files."""
        super().__init__(**kwargs)
        self._load_agent_config()
        self._load_models_config()
        self._load_memory_config()
        self._load_api_config()
        self._apply_memory_config()
        self._apply_api_config()
        self._ensure_directories()

    def _load_agent_config(self) -> None:
        """Load agent configuration from YAML."""
        config_path = self.config_dir / "agents_config.yaml"
        if config_path.exists():
            with open(config_path, "r") as f:
                self.agents_config = yaml.safe_load(f)

    def _load_models_config(self) -> None:
        """Load models configuration from YAML."""
        config_path = self.config_dir / "models_config.yaml"
        if config_path.exists():
            with open(config_path, "r") as f:
                self.models_config = yaml.safe_load(f)

    def _load_memory_config(self) -> None:
        """Load memory configuration from YAML."""
        config_path = self.config_dir / "memory_config.yaml"
        if config_path.exists():
            with open(config_path, "r") as f:
                self.memory_config = yaml.safe_load(f)

    def _load_api_config(self) -> None:
        """Load API configuration from YAML."""
        config_path = self.config_dir / "api_config.yaml"
        if config_path.exists():
            with open(config_path, "r") as f:
                self.api_config = yaml.safe_load(f)

    def _apply_memory_config(self) -> None:
        """Apply memory configuration from YAML to memory settings."""
        if not self.memory_config:
            return

        # Get current environment
        env = self.system.environment

        # Start with base memory config
        mem_config = self.memory_config.get("memory", {})

        # Apply environment-specific overrides
        env_overrides = self.memory_config.get("environments", {}).get(env, {})

        # Helper to merge nested configs
        def merge_config(base: dict, override: dict) -> dict:
            merged = base.copy()
            for key, value in override.items():
                if (
                    isinstance(value, dict)
                    and key in merged
                    and isinstance(merged[key], dict)
                ):
                    merged[key] = merge_config(merged[key], value)
                else:
                    merged[key] = value
            return merged

        # Merge environment-specific settings
        if env_overrides:
            mem_config = merge_config(mem_config, env_overrides)

        # Apply to memory settings object
        if "storage" in mem_config:
            storage = mem_config["storage"]
            self.memory.storage_dir = storage.get("base_dir", self.memory.storage_dir)
            self.memory.execution_records_dir = storage.get(
                "execution_records_dir", self.memory.execution_records_dir
            )
            self.memory.vector_store_dir = storage.get(
                "vector_store_dir", self.memory.vector_store_dir
            )
            self.memory.vector_store_name = storage.get(
                "vector_store_name", self.memory.vector_store_name
            )

        if "backend" in mem_config:
            backend = mem_config["backend"]
            backend_type = backend.get("type", "redis")
            self.memory.backend_type = backend_type
            self.memory.use_redis = backend_type == "redis"

        if "ttl" in mem_config:
            ttl = mem_config["ttl"]
            self.memory.short_term_ttl = ttl.get(
                "short_term", self.memory.short_term_ttl
            )
            self.memory.session_timeout = ttl.get(
                "session_timeout", self.memory.session_timeout
            )
            self.memory.search_cache_ttl = ttl.get(
                "search_cache", self.memory.search_cache_ttl
            )

        if "session" in mem_config:
            session = mem_config["session"]
            self.memory.auto_cleanup = session.get(
                "auto_cleanup", self.memory.auto_cleanup
            )
            self.memory.max_conversation_history = session.get(
                "max_conversation_history", self.memory.max_conversation_history
            )
            self.memory.max_session_duration = session.get(
                "max_duration", self.memory.max_session_duration
            )
            self.memory.persist_sessions = session.get(
                "persist_sessions", self.memory.persist_sessions
            )

        if "long_term" in mem_config:
            lt = mem_config["long_term"]
            self.memory.enable_vector_search = lt.get(
                "enable_vector_search", self.memory.enable_vector_search
            )
            self.memory.embedding_model = lt.get(
                "embedding_model", self.memory.embedding_model
            )
            self.memory.embedding_dimensions = lt.get(
                "embedding_dimensions", self.memory.embedding_dimensions
            )
            self.memory.embedding_batch_size = lt.get(
                "embedding_batch_size", self.memory.embedding_batch_size
            )

            if "vector_search" in lt:
                vs = lt["vector_search"]
                self.memory.vector_search_top_k = vs.get(
                    "default_top_k", self.memory.vector_search_top_k
                )
                self.memory.similarity_threshold = vs.get(
                    "similarity_threshold", self.memory.similarity_threshold
                )
                self.memory.enable_reranking = vs.get(
                    "enable_reranking", self.memory.enable_reranking
                )

        if "performance" in mem_config:
            perf = mem_config["performance"]
            self.memory.cache_enabled = perf.get(
                "cache_enabled", self.memory.cache_enabled
            )
            self.memory.max_cache_size = perf.get(
                "max_cache_size", self.memory.max_cache_size
            )
            self.memory.cache_policy = perf.get(
                "cache_policy", self.memory.cache_policy
            )
            self.memory.lazy_load_vector_store = perf.get(
                "lazy_load_vector_store", self.memory.lazy_load_vector_store
            )

        if "execution_records" in mem_config:
            er = mem_config["execution_records"]
            self.memory.execution_records_enabled = er.get(
                "enabled", self.memory.execution_records_enabled
            )
            self.memory.execution_records_format = er.get(
                "format", self.memory.execution_records_format
            )
            self.memory.max_records_per_file = er.get(
                "max_records_per_file", self.memory.max_records_per_file
            )
            self.memory.retention_days = er.get(
                "retention_days", self.memory.retention_days
            )

        if "monitoring" in mem_config:
            mon = mem_config["monitoring"]
            self.memory.monitoring_enabled = mon.get(
                "enabled", self.memory.monitoring_enabled
            )
            self.memory.stats_interval = mon.get(
                "stats_interval", self.memory.stats_interval
            )
            if "alerts" in mon:
                alerts = mon["alerts"]
                self.memory.alert_short_term_high = alerts.get(
                    "short_term_high", self.memory.alert_short_term_high
                )
                self.memory.alert_active_sessions_high = alerts.get(
                    "active_sessions_high", self.memory.alert_active_sessions_high
                )

    def _apply_api_config(self) -> None:
        """Apply API configuration from YAML to FastAPI settings."""
        if not self.api_config:
            return

        # Get current environment
        env = self.system.environment

        # Start with base API config
        api_config = self.api_config.get("fastapi", {})

        # Apply environment-specific overrides
        env_overrides = self.api_config.get("environments", {}).get(env, {})
        if env_overrides and "fastapi" in env_overrides:
            for key, value in env_overrides["fastapi"].items():
                api_config[key] = value

        # Apply to FastAPI settings
        if "host" in api_config:
            self.fastapi.host = api_config["host"]
        if "port" in api_config:
            self.fastapi.port = api_config["port"]
        if "reload" in api_config:
            self.fastapi.reload = api_config["reload"]
        if "workers" in api_config:
            self.fastapi.workers = api_config["workers"]
        if "log_level" in api_config:
            self.fastapi.log_level = api_config["log_level"]

        # Apply CORS settings
        if "cors" in api_config:
            cors = api_config["cors"]
            self.fastapi.cors_enabled = cors.get("enabled", self.fastapi.cors_enabled)

        # Apply rate limiting
        if "rate_limit" in api_config:
            rl = api_config["rate_limit"]
            self.fastapi.rate_limit_enabled = rl.get(
                "enabled", self.fastapi.rate_limit_enabled
            )
            if "requests_per_minute" in rl:
                self.fastapi.rate_limit_requests = rl["requests_per_minute"]

        # Apply timeouts
        if "timeouts" in api_config:
            timeouts = api_config["timeouts"]
            if "request" in timeouts:
                self.fastapi.request_timeout = timeouts["request"]

        # Apply authentication settings from environment overrides
        if env_overrides and "authentication" in env_overrides:
            auth = env_overrides["authentication"]
            self.security.authentication_required = auth.get(
                "enabled", self.security.authentication_required
            )

    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        directories = [
            self.data_dir,
            self.logs_dir,
            self.data_dir / "embeddings",
            self.data_dir / "knowledge_base",
            self.data_dir / "processed",
            self.data_dir / "processed" / "memory",
            self.data_dir / "processed" / "memory" / "execution_records",
            self.logs_dir / "agents",
            self.logs_dir / "api",
            self.logs_dir / "evaluation",
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get_agent_config(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific agent."""
        if self.agents_config:
            return self.agents_config.get("agents", {}).get(agent_id)
        return None

    def get_model_config(
        self, provider: str, model_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific model."""
        if self.models_config:
            return self.models_config.get(provider, {}).get(model_name)
        return None

    def get_workflow_config(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific workflow."""
        if self.api_config:
            return self.api_config.get("workflows", {}).get(workflow_id)
        return None

    def get_endpoint_config(self, endpoint_category: str) -> Optional[Dict[str, Any]]:
        """Get configuration for API endpoints."""
        if self.api_config:
            return self.api_config.get("endpoints", {}).get(endpoint_category)
        return None


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings: Application settings
    """
    return Settings()
