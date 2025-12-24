"""
Pydantic-based configuration validation with runtime checks
"""

from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings

load_dotenv()


class AzureOpenAIConfig(BaseModel):
    """Azure OpenAI configuration with validation"""

    model_config = {"frozen": True}  # Immutable after creation

    # LLM Configuration
    llm_endpoint: str = Field(
        ..., min_length=1, description="Azure OpenAI endpoint URL"
    )
    llm_api_key: str = Field(..., min_length=1, description="Azure OpenAI API key")
    llm_deplyment_name: str = Field(
        ..., min_length=1, description="LLM deployment name"
    )
    llm_api_version: str = Field(
        default="2024-08-01-preview", description="API version"
    )
    llm_model_name: str = Field(default="gpt-4o", description="Model name")

    @field_validator("llm_endpoint")
    @classmethod
    def validate_endpoint(cls, v: str) -> str:
        """Validate endpoint is a valid URL"""
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"Endpoint must start with http:// or https://, got: {v}")
        return v.rstrip("/")

    @field_validator("llm_api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate API key is not a placeholder"""
        if v.strip().lower() in ("", "your-api-key", "placeholder", "xxx", "none"):
            raise ValueError("API key appears to be a placeholder value")
        if len(v.strip()) < 20:
            raise ValueError("API key appears to be too short (minimum 20 characters)")
        return v


class AzureStorageConfig(BaseModel):
    """Azure Blob Storage configuration with validation"""

    model_config = {"frozen": True}

    account_name: str = Field(
        ..., min_length=3, max_length=24, description="Storage account name"
    )
    container_name: str = Field(
        ..., min_length=3, max_length=63, description="Container name"
    )
    connection_string: str = Field(..., min_length=1, description="Connection string")
    blob_url: str = Field(
        default="azure://auxeestorage.blob.core.windows.net/auxee-upload-files/",
        description="Base blob URL",
    )
    account_key: Optional[str] = Field(default=None, description="Storage account key")
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
        if "--" in v:
            raise ValueError("Container name cannot contain consecutive hyphens")
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


class AppConfig(BaseSettings):
    """
    Main application configuration with Pydantic validation.

    Automatically loads from environment variables.
    Environment variables should be prefixed with the field name.

    Example .env:
        azureOpenAIEndpoint=https://...
        azureOpenAIApiKey=...
        azure_storage_account_name=...
    """

    model_config = {
        "case_sensitive": False,
        "env_nested_delimiter": "__",  # For nested configs: AZURE_OPENAI__LLM_ENDPOINT
    }

    # Sub-configurations
    azure_openai: AzureOpenAIConfig
    azure_storage: AzureStorageConfig

    # Application settings
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

        This method manually constructs the config from specific env vars
        to match your existing naming convention.

        Returns:
            Validated AppConfig instance

        Raises:
            ValidationError: If configuration is invalid
        """
        import os

        # Build AzureOpenAIConfig from env
        azure_openai = AzureOpenAIConfig(
            llm_endpoint=os.getenv("azureOpenAIEndpoint", ""),
            llm_api_key=os.getenv("azureOpenAIApiKey", ""),
            llm_deplyment_name=os.getenv("azureOpenAIApiDeploymentName", ""),
            llm_api_version=os.getenv("azureOpenAIApiVersion", "2024-08-01-preview"),
        )

        # Build AzureStorageConfig from env
        azure_storage = AzureStorageConfig(
            account_name=os.getenv("azure_storage_account_name", ""),
            container_name=os.getenv("azure_storage_container_name", ""),
            connection_string=os.getenv("azure_storage_connection_string", ""),
            account_key=os.getenv("azure_storage_account_key"),
        )

        # Build AppConfig
        return cls(
            azure_openai=azure_openai,
            azure_storage=azure_storage,
            enable_debug=os.getenv("ENABLE_DEBUG", "false").lower() == "true",
        )

    def validate(self) -> bool:
        """
        Additional validation beyond Pydantic's automatic validation.

        Note: Most validation happens automatically via Pydantic validators.
        This method is kept for backwards compatibility.
        """
        print("[INFO] Pydantic configuration validation passed")
        return True


# Global config instance
_config: Optional[AppConfig] = None


def get_config(force_reload: bool = False) -> AppConfig:
    """
    Get the global configuration instance.

    Args:
        force_reload: Force reload configuration from environment

    Returns:
        Validated AppConfig instance

    Raises:
        ValidationError: If configuration is invalid
    """
    global _config

    if _config is None or force_reload:
        try:
            _config = AppConfig.from_env()
            print("[INFO] Configuration loaded and validated successfully")
        except Exception as e:
            print(f"[FATAL ERROR] Configuration validation failed: {e}")
            raise

    return _config


# Backwards compatibility class
class Config:
    """Legacy Config class for backwards compatibility"""

    _app_config = None

    @classmethod
    def _get_app_config(cls):
        if cls._app_config is None:
            cls._app_config = get_config()
        return cls._app_config

    @classmethod
    @property
    def llm_deplyment_name(cls):
        return cls._get_app_config().azure_openai.llm_deplyment_name

    @classmethod
    @property
    def llm_api_key(cls):
        return cls._get_app_config().azure_openai.llm_api_key

    @classmethod
    @property
    def llm_endpoint(cls):
        return cls._get_app_config().azure_openai.llm_endpoint

    @classmethod
    @property
    def llm_api_version(cls):
        return cls._get_app_config().azure_openai.llm_api_version

    @classmethod
    @property
    def azure_storage_connection_string(cls):
        return cls._get_app_config().azure_storage.connection_string

    @classmethod
    @property
    def azure_storage_account_name(cls):
        return cls._get_app_config().azure_storage.account_name

    @classmethod
    @property
    def azure_storage_container_name(cls):
        return cls._get_app_config().azure_storage.container_name

    @staticmethod
    def validate_env():
        return get_config().validate()
