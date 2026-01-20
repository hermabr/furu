def load_env() -> None:
    from dotenv import load_dotenv

    load_dotenv()


# Preserve previous behavior: attempt to load `.env` at import-time.
load_env()
