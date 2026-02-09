# DocGPT (WIP)

## Useful commands

```uv sync```: install deps  
```uv run python main.py```: run the Discord bot  

## Requirements
- docker (for PostgreSQL + MongoDB)
- python 3.11+
- uv
- Gemini API key (AI_GEMINI_APIKEY or GOOGLE_API_KEY)
- Discord bot token (APP_DISCORD_TOKEN)

Vector storage uses `langchain-postgres` with psycopg3. Set `STORAGE_VECTOR_URL` in `.env` to match your PostgreSQL. For the project's `docker compose`, use:
```
STORAGE_VECTOR_URL=postgresql+psycopg://root:example@localhost:5432/postgres
```
If you get "password authentication failed", another PostgreSQL may be on port 5432—use `STORAGE_VECTOR_URL` with the correct credentials, or stop the other service and run `docker compose up`.

## How to use
Ask about the R data.table package documentation and contribution guide.

1. Set envars in .env file, use the .env.example file as an example;
2. Run ```docker compose up```
3. Ingest data (once): ```uv run python main.py --ingest```
4. Start the Discord bot: ```uv run python main.py```
