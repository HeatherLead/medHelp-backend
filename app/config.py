from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    mongoDB_url: str

    class Config:
        env_file = ".env"


settings = Settings()
