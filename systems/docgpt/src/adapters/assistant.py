from functools import lru_cache

from dependency_injector.providers import Factory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory.chat_memory import BaseChatMemory
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from src.core.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from src.domain.assistant import Message, SessionId
from src.port.assistant import AssistantPort


class ConversationalAssistantAdapter(AssistantPort):
    def __init__(
        self,
        llm: BaseChatModel,
        storage: VectorStore,
        memory_factory: Factory[BaseChatMemory],
        *,
        k: int = 100,
        tokens_limit: int = 4_000,
        score_threshold: float | None = 0.9,
        distance_threshold: float | None = None,
    ) -> None:
        self._llm = llm

        self._storage = storage
        self._memory_factory = memory_factory
        self._k = k
        self._tokens_limit = tokens_limit
        self._score_threshold = score_threshold
        self._distance_threshold = distance_threshold

    @lru_cache
    def _get_memory(self, session_id: SessionId) -> BaseChatMemory:
        return self._memory_factory(chat_memory__session_id=session_id)

    def clear_history(self, session_id: SessionId) -> None:
        self._get_memory(session_id).clear()

    def prompt(self, message: Message, *, session_id: SessionId | None = None) -> PromptResult:
        memory = self._get_memory(session_id) if session_id else None

        # Build search_kwargs, only include non-None values
        search_kwargs = {"k": self._k}
        if self._score_threshold is not None:
            search_kwargs["score_threshold"] = self._score_threshold
        if self._distance_threshold is not None:
            search_kwargs["distance_threshold"] = self._distance_threshold

        qa = ConversationalRetrievalChain.from_llm(
            llm=self._llm,
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
            retriever=self._storage.as_retriever(
                search_type="similarity",
                search_kwargs=search_kwargs,
            ),
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
            get_chat_history=lambda v: v,
            memory=memory,
            verbose=True,
            return_source_documents=True,
            # max_tokens_limit disabled due to Gemini API compatibility issue
            # max_tokens_limit=self._tokens_limit,
        )

        qa_params = dict(question=message)
        if not memory:
            qa_params["chat_history"] = ""

        response = qa(qa_params)

        # Extract retrieved context from source documents
        source_docs = response.get("source_documents", [])
        retrieved_context = "\n\n---\n\n".join(
            doc.page_content for doc in source_docs
        )
        source_metadata = [
            doc.metadata for doc in source_docs if hasattr(doc, "metadata")
        ]

        return PromptResult(
            answer=response["answer"],
            retrieved_context=retrieved_context,
            source_metadata=source_metadata,
        )
