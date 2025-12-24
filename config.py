"""
Universal Configuration Module
Centralized configuration for the entire parquet-pipeline application.
All modules should import config from here instead of loading env vars directly.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class VectorDBType(Enum):
    """Supported vector database types"""

    CHROMADB = "chromadb"


@dataclass
class AzureOpenAIConfig:
    """Azure OpenAI configuration for LLM and embeddings"""

    # --- Non-default fields (Must be defined first) ---
    # LLM (GPT-4o) Configuration
    llm_endpoint: str
    llm_api_key: str
    llm_deployment_name: str

    # Embedding Configuration
    embedding_endpoint: str
    embedding_api_key: str
    embedding_deployment_name: str

    # --- Default fields (Must be defined last) ---
    llm_api_version: str = "2024-08-01-preview"
    llm_model_name: str = "gpt-4o"
    embedding_api_version: str = "2024-02-01"

    @classmethod
    def from_env(cls):
        """Load configuration from environment variables"""
        # LLM settings
        llm_endpoint = os.getenv("azureOpenAIEndpoint")
        llm_api_key = os.getenv("azureOpenAIApiKey")
        llm_deployment_name = os.getenv("azureOpenAIApiDeploymentName")
        llm_api_version = os.getenv("azureOpenAIApiVersion", "2024-08-01-preview")

        # Embedding settings
        embedding_resource = os.getenv("OPENAI_EMBEDDING_RESOURCE")
        # NOTE: embedding_endpoint is set conditionally, but the dataclass definition requires it to be supplied
        # The logic here makes it None if the resource is missing, which is validated below.
        embedding_endpoint = (
            f"https://{embedding_resource}.openai.azure.com/"
            if embedding_resource
            else None
        )
        embedding_api_key = os.getenv("OPENAI_EMBEDDING_API_KEY")
        embedding_deployment_name = os.getenv("OPENAI_EMBEDDING_MODEL")
        embedding_api_version = os.getenv("OPENAI_EMBEDDING_VERSION", "2024-02-01")

        # Validate required fields
        if not all([llm_endpoint, llm_api_key, llm_deployment_name]):
            raise ValueError(
                "Missing required LLM configuration. Check your .env file."
            )

        if not all([embedding_endpoint, embedding_api_key, embedding_deployment_name]):
            raise ValueError(
                "Missing required embedding configuration. Check your .env file."
            )

        return cls(
            llm_endpoint=llm_endpoint,
            llm_api_key=llm_api_key,
            llm_deployment_name=llm_deployment_name,
            llm_api_version=llm_api_version,
            embedding_endpoint=embedding_endpoint,
            embedding_api_key=embedding_api_key,
            embedding_deployment_name=embedding_deployment_name,
            embedding_api_version=embedding_api_version,
        )


@dataclass
class AzureStorageConfig:
    """Azure Blob Storage configuration"""

    # Non-default fields
    account_name: str
    container_name: str
    connection_string: str
    blob_url: str = "azure://auxeestorage.blob.core.windows.net/auxee-upload-files/"

    # Default fields
    account_key: Optional[str] = None
    parquet_output_dir: str = "parquet_files/"

    @property
    def glob_pattern(self) -> str:
        """Generate glob pattern for all parquet files"""
        return (
            f"azure://{self.account_name}.blob.core.windows.net/"
            f"{self.container_name}/{self.parquet_output_dir}*.parquet"
        )

    @classmethod
    def from_env(cls):
        """Load configuration from environment variables"""
        account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
        container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        account_key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")

        if not all([account_name, container_name, connection_string]):
            raise ValueError(
                "Missing required Azure Storage configuration. Check your .env file."
            )

        return cls(
            account_name=account_name,
            container_name=container_name,
            connection_string=connection_string,
            account_key=account_key,
        )


@dataclass
class ChromaDBConfig:
    """ChromaDB configuration"""

    # Default fields
    persist_directory: str = "./chroma_db"
    collection_prefix: str = "data_source"
    anonymized_telemetry: bool = False

    @classmethod
    def from_env(cls):
        """Load configuration from environment variables"""
        persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
        collection_prefix = os.getenv("CHROMA_COLLECTION_PREFIX", "data_source")

        return cls(
            persist_directory=persist_directory, collection_prefix=collection_prefix
        )


@dataclass
class VectorDBConfig:
    """Vector database configuration"""

    # Non-default fields
    db_type: VectorDBType

    # Default fields
    chromadb: Optional[ChromaDBConfig] = None

    @classmethod
    def from_env(cls, db_type: VectorDBType = VectorDBType.CHROMADB):
        """Load configuration based on selected database type"""

        db_type = VectorDBType.CHROMADB

        # Default to ChromaDB
        chromadb_config = ChromaDBConfig.from_env()
        return cls(db_type=db_type, chromadb=chromadb_config)


@dataclass
class AppConfig:
    """Main application configuration"""

    # Non-default fields (Sub-configurations)
    azure_openai: AzureOpenAIConfig
    azure_storage: AzureStorageConfig
    vector_db: VectorDBConfig

    # Default fields
    # Application settings
    input_file_paths: List[str] = field(
        default_factory=lambda: [
            "../sheets/file1.xlsx",
            "../sheets/file2.xlsx",
            "../sheets/loan.xlsx",
            "../sheets/Formulation_Test.xlsx",
            "../sheets/Formulation2.xlsx",
            "../sheets/PAID NARCAN Sample Data.xlsx",
        ]
    )

    # Performance settings
    enable_debug: bool = False

    @classmethod
    def from_env(cls, vector_db_type: VectorDBType = VectorDBType.CHROMADB):
        """Load complete configuration from environment variables"""
        try:
            azure_openai = AzureOpenAIConfig.from_env()
            azure_storage = AzureStorageConfig.from_env()
            vector_db = VectorDBConfig.from_env(db_type=vector_db_type)

            print("[INFO] Configuration loaded successfully")

            return cls(
                azure_openai=azure_openai,
                azure_storage=azure_storage,
                vector_db=vector_db,
            )
        except ValueError as e:
            print(f"[FATAL ERROR] Configuration failed: {e}")
            raise

    def validate(self) -> bool:
        """Validate configuration"""
        # Check if all required settings are present
        if not self.azure_openai.llm_api_key:
            print("[ERROR] Missing LLM API key")
            return False

        if not self.azure_openai.embedding_api_key:
            print("[ERROR] Missing Embedding API key")
            return False

        if not self.azure_storage.connection_string:
            print("[ERROR] Missing Azure Storage connection string")
            return False

        print("[INFO] Configuration validation passed")
        return True


# Global config instance
_config: Optional[AppConfig] = None


def get_config(
    vector_db_type: VectorDBType = VectorDBType.CHROMADB, force_reload: bool = False
) -> AppConfig:
    """
    Get the global configuration instance.

    Args:
        vector_db_type: Type of vector database to use (default: ChromaDB)
        force_reload: Force reload configuration from environment

    Returns:
        AppConfig instance
    """
    global _config

    if _config is None or force_reload:
        _config = AppConfig.from_env(vector_db_type=vector_db_type)
        _config.validate()

    return _config


def set_vector_db(db_type: VectorDBType):
    """Switch vector database type"""
    global _config
    _config = AppConfig.from_env(vector_db_type=db_type)
    _config.validate()
    print(f"[INFO] Switched to {db_type.value}")


# For backwards compatibility with existing Config class
class Config:
    """
    Legacy Config class for backwards compatibility.
    New code should use get_config() instead.
    """

    _app_config = None

    @classmethod
    def _get_app_config(cls):
        if cls._app_config is None:
            cls._app_config = get_config()
        return cls._app_config

    @classmethod
    @property
    def LLM_DEPLOYMENT_NAME(cls):
        return cls._get_app_config().azure_openai.llm_deployment_name

    @classmethod
    @property
    def LLM_API_KEY(cls):
        return cls._get_app_config().azure_openai.llm_api_key

    @classmethod
    @property
    def LLM_ENDPOINT(cls):
        return cls._get_app_config().azure_openai.llm_endpoint

    @classmethod
    @property
    def LLM_API_VERSION(cls):
        return cls._get_app_config().azure_openai.llm_api_version

    @classmethod
    @property
    def LLM_MODEL_NAME(cls):
        return cls._get_app_config().azure_openai.llm_model_name

    @classmethod
    @property
    def AZURE_STORAGE_ACCOUNT_NAME(cls):
        return cls._get_app_config().azure_storage.account_name

    @classmethod
    @property
    def AZURE_STORAGE_CONTAINER_NAME(cls):
        return cls._get_app_config().azure_storage.container_name

    @classmethod
    @property
    def AZURE_STORAGE_ACCOUNT_KEY(cls):
        return cls._get_app_config().azure_storage.account_key

    @classmethod
    @property
    def AZURE_STORAGE_CONNECTION_STRING(cls):
        return cls._get_app_config().azure_storage.connection_string

    @classmethod
    @property
    def INPUT_FILE_PATHS(cls):
        return cls._get_app_config().input_file_paths

    @classmethod
    @property
    def PARQUET_OUTPUT_DIR(cls):
        return cls._get_app_config().azure_storage.parquet_output_dir

    @classmethod
    @property
    def ALL_PARQUET_GLOB_PATTERN(cls):
        return cls._get_app_config().azure_storage.glob_pattern

    @staticmethod
    def validate_env():
        """Validate environment configuration"""
        return get_config().validate()
