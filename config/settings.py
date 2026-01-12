"""
Centralized Settings and Configuration Management.

Loads configuration from environment variables, YAML files, and provides
validated settings for the entire application.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List
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


class SecuritySettings(BaseSettings):
    """Security configuration."""

    authentication_required: bool = Field(default=True, env="AUTHENTICATION_REQUIRED")
    api_key_rotation_days: int = Field(default=90, env="API_KEY_ROTATION_DAYS")
    max_requests_per_minute: int = Field(default=100, env="MAX_REQUESTS_PER_MINUTE")
    allowed_origins: List[str] = Field(
        default=["localhost", "127.0.0.1"], env="ALLOWED_ORIGINS"
    )


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
    system: SystemSettings = Field(default_factory=SystemSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)

    # Agent configuration
    agents_config: Optional[Dict[str, Any]] = None
    models_config: Optional[Dict[str, Any]] = None

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="allow"
    )

    def __init__(self, **kwargs):
        """Initialize settings and load configuration files."""
        super().__init__(**kwargs)
        self._load_agent_config()
        self._load_models_config()
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

    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        directories = [
            self.data_dir,
            self.logs_dir,
            self.data_dir / "embeddings",
            self.data_dir / "knowledge_base",
            self.data_dir / "processed",
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


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings: Application settings
    """
    return Settings()
