import logging
import re
from functools import lru_cache


logger = logging.getLogger(__name__)

FALLBACK_STOPWORDS = {"can", "please", "show", "tell"}


def _ensure_nltk_resource(resource: str, download_name: str) -> None:
    import nltk

    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(download_name, quiet=True)


@lru_cache(maxsize=8)
def _load_nltk_stopwords(language: str) -> set[str]:
    _ensure_nltk_resource("corpora/stopwords", "stopwords")
    from nltk.corpus import stopwords

    try:
        return set(stopwords.words(language))
    except OSError:
        logger.warning("query_preprocessor.stopwords language=%s unavailable", language)
        return set()


@lru_cache(maxsize=1)
def _load_wordnet():
    _ensure_nltk_resource("corpora/wordnet", "wordnet")
    _ensure_nltk_resource("corpora/omw-1.4", "omw-1.4")
    from nltk.corpus import wordnet as wn

    return wn


def build_retrieval_query(query: str, cfg=None) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", query.lower())
    if not tokens:
        return ""

    preprocessing = getattr(getattr(cfg, "retrieval", None), "preprocessing", None) if cfg else None
    language = getattr(preprocessing, "language", "english") if preprocessing else "english"
    use_nltk_stopwords = getattr(preprocessing, "use_nltk_stopwords", True) if preprocessing else True
    expand_synonyms = getattr(preprocessing, "expand_synonyms", True) if preprocessing else True
    max_synonyms_per_term = getattr(preprocessing, "max_synonyms_per_term", 3) if preprocessing else 3
    min_token_length = getattr(preprocessing, "min_token_length", 2) if preprocessing else 2
    stopwords = set(FALLBACK_STOPWORDS)
    if use_nltk_stopwords:
        stopwords.update(_load_nltk_stopwords(language))
    expansions = _normalize_expansions(getattr(preprocessing, "expansions", {})) if preprocessing else {}

    processed: list[str] = []
    seen: set[str] = set()

    for token in tokens:
        if token in stopwords:
            continue
        candidates = [token, *expansions.get(token, [])]
        if expand_synonyms:
            candidates.extend(_wordnet_expansions(token, max_synonyms_per_term))
        for candidate in candidates:
            if candidate in stopwords:
                continue
            if len(candidate) < min_token_length:
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            processed.append(candidate)

    return " ".join(processed) if processed else " ".join(tokens)


def _normalize_expansions(raw) -> dict[str, list[str]]:
    if not raw:
        return {}
    if hasattr(raw, "__dict__"):
        raw = raw.__dict__

    normalized: dict[str, list[str]] = {}
    for key, value in raw.items():
        if hasattr(value, "__dict__"):
            value = value.__dict__
        if isinstance(value, (list, tuple)):
            normalized[str(key).lower()] = [str(item).lower() for item in value]
    return normalized


@lru_cache(maxsize=512)
def _wordnet_expansions(token: str, limit: int) -> tuple[str, ...]:
    if len(token) < 4:
        return ()

    wn = _load_wordnet()
    expansions: list[str] = []
    seen: set[str] = set()

    synsets = wn.synsets(token)
    if not synsets:
        return ()

    for synset in synsets[:1]:
        for lemma in synset.lemma_names():
            normalized = lemma.replace("_", " ").lower()
            for part in normalized.split():
                if not part.isalpha():
                    continue
                if part == token or part in seen:
                    continue
                seen.add(part)
                expansions.append(part)
                if len(expansions) >= limit:
                    return tuple(expansions)

    return tuple(expansions)
