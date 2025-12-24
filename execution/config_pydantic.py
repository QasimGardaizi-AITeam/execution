"""
Pydantic-based configuration validation (requires pydantic to be installed)

To use this configuration:
1. Install pydantic: pip install pydantic
2. Replace imports in your code:
   from pipeline.config_pydantic import get_config

This provides:
- Automatic validation of all configuration values
- Type checking at runtime
- Clear error messages for missing/invalid config
- Environment variable parsing with defaults
"""

from enum import Enum
from typing import List, Optional

try:
    from pydantic import BaseModel, ConfigDict, Field, field_validator
    from pydantic_settings import BaseSettings

    PYDANTIC_AVAILABLE = True
except ImportError:
    # Fallback for when pydantic is not installed
    PYDANTIC_AVAILABLE = False
    BaseModel = object
    BaseSettings = object
    Field = lambda *args, **kwargs: None


class VectorDBType(str, Enum):
    """Supported vector database types"""

    CHROMADB = "chromadb"


if PYDANTIC_AVAILABLE:

    class AzureOpenAIConfig(BaseModel):
        """Azure OpenAI configuration with validation"""

        model_config = ConfigDict(frozen=True)

        # LLM Configuration
        llm_endpoint: str = Field(
            ..., min_length=1, description="Azure OpenAI endpoint URL"
        )
        llm_api_key: str = Field(..., min_length=1, description="Azure OpenAI API key")
        llm_deployment_name: str = Field(
            ..., min_length=1, description="LLM deployment name"
        )
        llm_api_version: str = Field(
            default="2024-08-01-preview", description="API version"
        )
        llm_model_name: str = Field(default="gpt-4o", description="Model name")

        # Embedding Configuration
        embedding_endpoint: str = Field(
            ..., min_length=1, description="Embedding endpoint URL"
        )
        embedding_api_key: str = Field(
            ..., min_length=1, description="Embedding API key"
        )
        embedding_deployment_name: str = Field(
            ..., min_length=1, description="Embedding deployment"
        )
        embedding_api_version: str = Field(
            default="2024-02-01", description="Embedding API version"
        )

        @field_validator("llm_endpoint", "embedding_endpoint")
        @classmethod
        def validate_endpoint(cls, v: str) -> str:
            """Validate endpoint is a valid URL"""
            if not v.startswith(("http://", "https://")):
                raise ValueError(
                    f"Endpoint must start with http:// or https://, got: {v}"
                )
            return v.rstrip("/")

        @field_validator("llm_api_key", "embedding_api_key")
        @classmethod
        def validate_api_key(cls, v: str) -> str:
            """Validate API key is not empty or placeholder"""
            if v.strip() in ("", "your-api-key", "placeholder", "xxx"):
                raise ValueError("API key appears to be a placeholder value")
            return v

    class AzureStorageConfig(BaseModel):
        """Azure Blob Storage configuration with validation"""

        model_config = ConfigDict(frozen=True)

        account_name: str = Field(
            ..., min_length=3, max_length=24, description="Storage account name"
        )
        container_name: str = Field(
            ..., min_length=3, max_length=63, description="Container name"
        )
        connection_string: str = Field(
            ..., min_length=1, description="Connection string"
        )
        blob_url: str = Field(
            default="azure://auxeestorage.blob.core.windows.net/auxee-upload-files/",
            description="Base blob URL",
        )
        account_key: Optional[str] = Field(
            default=None, description="Storage account key"
        )
        parquet_output_dir: str = Field(
            default="parquet_files/", description="Parquet output directory"
        )

        @field_validator("account_name")
        @classmethod
        def validate_account_name(cls, v: str) -> str:
            """Validate storage account name follows Azure naming rules"""
            if not v.islower():
                raise ValueError("Storage account name must be lowercase")
            if not v.isalnum():
                raise ValueError("Storage account name must be alphanumeric")
            return v

        @field_validator("container_name")
        @classmethod
        def validate_container_name(cls, v: str) -> str:
            """Validate container name follows Azure naming rules"""
            if not v.islower():
                raise ValueError("Container name must be lowercase")
            if not all(c.isalnum() or c == "-" for c in v):
                raise ValueError(
                    "Container name can only contain lowercase letters, numbers, and hyphens"
                )
            if v.startswith("-") or v.endswith("-"):
                raise ValueError("Container name cannot start or end with a hyphen")
            return v

        @field_validator("connection_string")
        @classmethod
        def validate_connection_string(cls, v: str) -> str:
            """Validate connection string format"""
            required_parts = ["AccountName=", "AccountKey="]
            if not all(part in v for part in required_parts):
                raise ValueError(
                    "Connection string must contain AccountName and AccountKey"
                )
            return v

        @property
        def glob_pattern(self) -> str:
            """Generate glob pattern for all parquet files"""
            return (
                f"azure://{self.account_name}.blob.core.windows.net/"
                f"{self.container_name}/{self.parquet_output_dir}*.parquet"
            )

    class ChromaDBConfig(BaseModel):
        """ChromaDB configuration"""

        model_config = ConfigDict(frozen=True)

        persist_directory: str = Field(
            default="./chroma_db", description="Persistence directory"
        )
        collection_prefix: str = Field(
            default="data_source", description="Collection name prefix"
        )
        anonymized_telemetry: bool = Field(
            default=False, description="Enable telemetry"
        )

    class VectorDBConfig(BaseModel):
        """Vector database configuration"""

        model_config = ConfigDict(frozen=True)

        db_type: VectorDBType = Field(
            default=VectorDBType.CHROMADB, description="Database type"
        )
        chromadb: Optional[ChromaDBConfig] = Field(
            default=None, description="ChromaDB config"
        )

    class AppConfig(BaseSettings):
        """
        Main application configuration with Pydantic validation.

        Automatically loads from environment variables with PIPELINE_ prefix.
        """

        model_config = ConfigDict(
            env_prefix="PIPELINE_",
            env_nested_delimiter="__",
            case_sensitive=False,
        )

        # Sub-configurations
        azure_openai: AzureOpenAIConfig
        azure_storage: AzureStorageConfig
        vector_db: VectorDBConfig = Field(
            default_factory=lambda: VectorDBConfig(
                db_type=VectorDBType.CHROMADB, chromadb=ChromaDBConfig()
            )
        )

        # Application settings
        input_file_paths: List[str] = Field(
            default_factory=lambda: [
                "../sheets/file1.xlsx",
                "../sheets/file2.xlsx",
                "../sheets/loan.xlsx",
                "../sheets/Formulation_Test.xlsx",
                "../sheets/Formulation2.xlsx",
                "../sheets/PAID NARCAN Sample Data.xlsx",
            ],
            description="Input file paths",
        )

        enable_debug: bool = Field(default=False, description="Enable debug mode")
        max_retries: int = Field(
            default=3, ge=1, le=10, description="Maximum retry attempts"
        )
        timeout_seconds: int = Field(
            default=30, ge=5, le=300, description="Request timeout"
        )

        @classmethod
        def from_env(cls) -> "AppConfig":
            """
            Load configuration from environment variables.

            Returns:
                Validated AppConfig instance

            Raises:
                ValidationError: If configuration is invalid
            """
            # This will automatically load from environment variables
            # and validate all fields
            return cls()

    # Global config instance
    _config: Optional[AppConfig] = None

    def get_config(force_reload: bool = False) -> AppConfig:
        """
        Get the global configuration instance.

        Args:
            force_reload: Force reload configuration from environment

        Returns:
            Validated AppConfig instance
        """
        global _config

        if _config is None or force_reload:
            _config = AppConfig.from_env()

        return _config

else:
    # Pydantic not available - provide helpful error message
    def get_config(force_reload: bool = False):
        raise ImportError(
            "Pydantic is not installed. Please install it with: pip install pydantic pydantic-settings"
        )
