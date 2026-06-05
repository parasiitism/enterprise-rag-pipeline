from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from enterprise_rag.embeddings.models import (
    EmbeddingProviderInfo,
    EmbeddingRequest,
    EmbeddingResult,
)


DEFAULT_HUGGINGFACE_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_HUGGINGFACE_EMBEDDING_DIMENSIONS = 384
DEFAULT_HUGGINGFACE_MAX_BATCH_SIZE = 32

_TLS_CONFIGURED = False


class SentenceTransformerLike(Protocol):
    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> Any:
        raise NotImplementedError


class SentenceTransformerLoader(Protocol):
    def __call__(self, model_name: str, **model_kwargs: object) -> SentenceTransformerLike:
        raise NotImplementedError


class HuggingFaceEmbeddingProvider:
    def __init__(
        self,
        model_name: str = DEFAULT_HUGGINGFACE_EMBEDDING_MODEL,
        dimensions: int = DEFAULT_HUGGINGFACE_EMBEDDING_DIMENSIONS,
        max_batch_size: int = DEFAULT_HUGGINGFACE_MAX_BATCH_SIZE,
        normalize_embeddings: bool = True,
        device: str | None = None,
        model: SentenceTransformerLike | None = None,
        model_loader: SentenceTransformerLoader | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must be a non-empty string")

        if dimensions <= 0:
            raise ValueError("dimensions must be greater than 0")

        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be greater than 0")

        self._info = EmbeddingProviderInfo(
            provider="huggingface",
            model=model_name,
            dimensions=dimensions,
            max_batch_size=max_batch_size,
        )
        self._normalize_embeddings = normalize_embeddings
        self._device = device
        self._model = model
        self._model_loader = model_loader

    @property
    def info(self) -> EmbeddingProviderInfo:
        return self._info

    def embed_texts(
        self,
        requests: Sequence[EmbeddingRequest],
    ) -> list[EmbeddingResult]:
        if not requests:
            return []

        texts = [request.text for request in requests]
        raw_embeddings = self._load_model().encode(
            texts,
            batch_size=min(len(texts), self.info.max_batch_size),
            normalize_embeddings=self._normalize_embeddings,
            show_progress_bar=False,
        )

        vectors = _to_float_vectors(raw_embeddings)

        if len(vectors) != len(requests):
            raise ValueError("embedding result count does not match request count")

        return [
            self._to_embedding_result(request=request, vector=vector)
            for request, vector in zip(requests, vectors, strict=True)
        ]

    def _load_model(self) -> SentenceTransformerLike:
        if self._model is not None:
            return self._model

        _configure_huggingface_tls()

        model_kwargs: dict[str, object] = {}

        if self._device is not None:
            model_kwargs["device"] = self._device

        loader = self._model_loader or _load_sentence_transformer_loader()

        try:
            self._model = loader(self.info.model, **model_kwargs)
        except RuntimeError as exc:
            if not _is_closed_http_client_error(exc):
                raise _model_load_error(self.info.model) from exc

            _reset_huggingface_session()

            try:
                self._model = loader(self.info.model, **model_kwargs)
            except Exception as retry_exc:
                raise _model_load_error(self.info.model) from retry_exc
        except Exception as exc:
            raise _model_load_error(self.info.model) from exc

        return self._model

    def _to_embedding_result(
        self,
        request: EmbeddingRequest,
        vector: list[float],
    ) -> EmbeddingResult:
        if len(vector) != self.info.dimensions:
            raise ValueError(
                f"expected embedding dimension {self.info.dimensions}, got {len(vector)}"
            )

        return EmbeddingResult(
            record_id=request.record_id,
            vector=vector,
            provider=self.info.provider,
            model=self.info.model,
            dimensions=self.info.dimensions,
            metadata=request.metadata,
        )


def _to_float_vectors(raw_embeddings: Any) -> list[list[float]]:
    if hasattr(raw_embeddings, "tolist"):
        raw_embeddings = raw_embeddings.tolist()

    return [
        [float(value) for value in vector]
        for vector in raw_embeddings
    ]


def _load_sentence_transformer_loader() -> SentenceTransformerLoader:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for HuggingFaceEmbeddingProvider. "
            'Install it with: python -m pip install -e ".[huggingface]"'
        ) from exc

    return SentenceTransformer


def _configure_huggingface_tls() -> None:
    global _TLS_CONFIGURED

    if _TLS_CONFIGURED:
        return

    try:
        import ssl

        import httpx
        import truststore
        from huggingface_hub.utils import close_session, set_client_factory
    except ImportError:
        _TLS_CONFIGURED = True
        return

    def client_factory() -> httpx.Client:
        tls_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        return httpx.Client(verify=tls_context)

    set_client_factory(client_factory)
    close_session()
    _TLS_CONFIGURED = True


def _reset_huggingface_session() -> None:
    try:
        from huggingface_hub.utils import close_session
    except ImportError:
        return

    close_session()


def _is_closed_http_client_error(exc: RuntimeError) -> bool:
    return "client has been closed" in str(exc).lower()


def _model_load_error(model_name: str) -> RuntimeError:
    return RuntimeError(
        f"Failed to load Hugging Face embedding model '{model_name}'. "
        'First run: python -m pip install -e ".[huggingface]". '
        "If the download still fails with SSL certificate errors, your network "
        "may be using a proxy or certificate authority that Python cannot trust."
    )
