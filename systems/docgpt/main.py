import logging
import os
from pathlib import Path

import pypandoc
from dependency_injector.wiring import Provide, inject
from dotenv import load_dotenv
from langchain_core.globals import set_debug, set_verbose
from langchain_core.vectorstores import VectorStore

from src.app.api import create_app, run_app
from src.app.discord import BOT
from src.core import containers
from src.core.interaction_logger import init_logger
from src.domain.content import Content
from src.port.assistant import AssistantPort
from src.port.content import ContentPort

logger = logging.getLogger(__name__)


@inject
def run_terminal(
    chat: AssistantPort = Provide[containers.Settings.assistant.chat],
):
    from src.core.interaction_logger import get_logger

    while True:
        question = input("-> **Q**: ")
        if question.lower() in ["q", "quit", "exit"]:
            break

        result = chat.prompt(question, session_id="cli")

        # Log the interaction
        interaction_logger = get_logger()
        if interaction_logger:
            try:
                interaction_logger.log(
                    session_id="cli",
                    question=question,
                    answer=result.answer,
                    retrieved_context=result.retrieved_context,
                    source_metadata=result.source_metadata,
                )
            except Exception:
                logger.exception("Failed to log interaction")

        print(f"**-> Q: {question}\n")
        print(f"**AI**: {result.answer}\n")


@inject
def run_discord(
    token: str = Provide[containers.Settings.app.discord_token],
):
    BOT.run(token)


@inject
def add_documents(
    documents: list[Content],
    *,
    storage: VectorStore = Provide[containers.Settings.storage.vector_storage],
) -> None:
    fails_count = 0
    failed_files = []

    for doc in documents:
        try:
            storage.add_documents([doc])
        except Exception as e:
            fails_count += 1

            # Extract file information from metadata
            metadata = doc.metadata if hasattr(doc, "metadata") else {}
            file_name = metadata.get("file_name", "Unknown")
            file_path = metadata.get("file_path", metadata.get("source", "Unknown"))
            project = metadata.get("project", "Unknown")
            source = metadata.get("source", "Unknown")

            # Determine file type from file extension
            file_type = "Unknown"
            if file_name and file_name != "Unknown":
                file_type = Path(file_name).suffix or "No extension"
            elif file_path and file_path != "Unknown":
                file_type = Path(file_path).suffix or "No extension"

            # Get exception details
            exception_type = type(e).__name__
            exception_message = str(e)

            # Log detailed error information
            logger.error(
                f"Failed to ingest file - "
                f"File Name: {file_name}, "
                f"File Type: {file_type}, "
                f"File Path: {file_path}, "
                f"Project: {project}, "
                f"Source: {source}, "
                f"Exception Type: {exception_type}, "
                f"Reason: {exception_message}"
            )

            failed_files.append(
                {
                    "file_name": file_name,
                    "file_type": file_type,
                    "file_path": file_path,
                    "project": project,
                    "source": source,
                    "exception_type": exception_type,
                    "reason": exception_message,
                }
            )

    if fails_count:
        logger.warning(f"Total of {fails_count} documents failed to ingest")
        logger.info(f"Failed files summary: {failed_files}")


@inject
def fetch_documents(
    code: ContentPort = Provide[containers.Settings.content.git_code],
    wiki: ContentPort = Provide[containers.Settings.content.git_wiki],
    assets_path: Path = Provide[containers.Settings.core.assets_path],
):
    project = "data.table"
    org = "Rdatatable"

    code_url = f"https://github.com/{org}/{project}.git"
    code_path = assets_path.joinpath(project)

    wiki_url = f"https://github.com/{org}/{project}.wiki.git"
    wiki_path = assets_path.joinpath(project + ".wiki")

    if wiki_path.exists():
        wiki_docs = wiki.get_by_path(project, wiki_path)
    else:
        wiki_docs = wiki.get_by_url(project, wiki_url)

    if code_path.exists():
        code_docs = code.get_by_path(project, code_path, branch="master")
    else:
        code_docs = code.get_by_url(project, code_url, branch="master")

    add_documents(wiki_docs)  # type: ignore
    add_documents(code_docs)  # type: ignore


def _parse_args() -> tuple[bool, bool]:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ingest data.table repo and wiki into vector storage",
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Run the FastAPI server instead of the Discord bot",
    )
    args = parser.parse_args()
    return args.ingest, args.api


@inject
def run_api(
    settings: containers.Settings = Provide[containers.Settings],
    port: int = Provide[containers.Settings.api.port],
) -> None:
    app = create_app(settings)
    run_app(app, port)


if __name__ == "__main__":
    load_dotenv()
    pypandoc.ensure_pandoc_installed()

    application = containers.Settings()
    application.config.from_yaml("config.yml", envs_required=True, required=True)
    application.core.init_resources()
    application.wire(
        modules=[
            __name__,
            "src.app.discord",
            "src.app.api.v1.endpoints.assistant",
        ]
    )
    set_debug(True)
    set_verbose(True)

    # Initialise the interaction logger — logs go to INTERACTION_LOG_DIR or ./logs
    log_dir = os.environ.get("INTERACTION_LOG_DIR", "logs")
    interaction_logger = init_logger(output_dir=log_dir)
    logger.info(
        "Interaction logs: CSV=%s, JSONL=%s",
        interaction_logger.csv_path,
        interaction_logger.jsonl_path,
    )

    do_ingest, run_api_mode = _parse_args()

    if do_ingest:
        fetch_documents()  # type: ignore

    if run_api_mode:
        run_api()  # type: ignore
    else:
        run_discord()  # type: ignore
