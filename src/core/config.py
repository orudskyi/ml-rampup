from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    GOOGLE_API_KEY: str
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "knowledge_base"
    

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

# Singleton instance
settings = Settings()