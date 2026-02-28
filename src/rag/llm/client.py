import json
import logging


logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        if not self.api_key:
            return None
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "openai is required for real generation. Reinstall the project dependencies."
                ) from exc
            self._client = OpenAI(api_key=self.api_key)
            logger.info("llm.client.initialized provider=openai")
        return self._client

    def _extract_response_text(self, response) -> str:
        output_text = getattr(response, "output_text", "") or ""
        if output_text.strip():
            return output_text.strip()

        parts: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text_value = getattr(content, "text", "")
                if isinstance(text_value, str) and text_value.strip():
                    parts.append(text_value.strip())
                elif hasattr(text_value, "value") and str(text_value.value).strip():
                    parts.append(str(text_value.value).strip())
        return "\n".join(parts).strip()

    def generate(
        self,
        system: str,
        user: str,
        context: list[str],
        temperature: float,
        model: str,
        max_output_tokens: int,
    ) -> str:
        if not context:
            logger.info("llm.generate.skipped reason=no_context")
            return "I do not know based on the available context."

        client = self._get_client()
        if client is None:
            logger.warning("llm.generate.skipped reason=missing_api_key")
            return "No LLM is configured. Add OPENAI_API_KEY to generate answers from retrieved context."

        context_text = "\n\n".join(context)
        logger.info(
            "llm.generate.start provider=openai model=%s context_blocks=%d",
            model,
            len(context),
        )
        request_payload = {
            "model": model,
            "instructions": system,
            "input": (
                "Answer the user's question using only the supplied context.\n\n"
                f"Context:\n{context_text}\n\n"
                f"Question: {user}"
            ),
            "max_output_tokens": max_output_tokens,
        }
        if not model.startswith("gpt-5"):
            request_payload["temperature"] = temperature

        response = client.responses.create(**request_payload)
        logger.info("llm.generate.complete provider=openai model=%s", model)
        text = self._extract_response_text(response)
        return text or "The model returned an empty response."

    def evaluate_answer(
        self,
        *,
        query: str,
        answer: str,
        retrieval_query: str | None,
        hits: list[dict[str, str | float | int | None]],
        model: str,
        max_output_tokens: int,
    ) -> dict[str, str]:
        client = self._get_client()
        if client is None:
            logger.warning("llm.evaluate.skipped reason=missing_api_key")
            return {
                "verdict": "unavailable",
                "reasoning": "No LLM is configured. Add OPENAI_API_KEY to evaluate answers.",
            }

        logger.info("llm.evaluate.start provider=openai model=%s", model)
        evidence_blocks = []
        for index, hit in enumerate(hits, start=1):
            evidence_blocks.append(
                "\n".join(
                    [
                        f"Evidence {index}",
                        f"doc_id: {hit.get('doc_id', '')}",
                        f"title: {hit.get('title', '')}",
                        f"score: {hit.get('score', '')}",
                        f"snippet: {hit.get('snippet', '')}",
                    ]
                )
            )
        evidence_text = "\n\n".join(evidence_blocks) if evidence_blocks else "No retrieved evidence provided."
        instructions = (
            "You are a RAG answer judge. Evaluate the answer against the question and the retrieved evidence.\n"
            "Judge three things together: relevance to the question, grounding in the evidence, and hallucination risk.\n"
            "Return a JSON object with exactly these keys:\n"
            '{"verdict":"match or mismatch","reasoning":"one short explanation"}'
        )
        request_payload = {
            "model": model,
            "instructions": instructions,
            "input": (
                f"Question: {query}\n"
                f"Retrieval query: {retrieval_query or 'n/a'}\n\n"
                f"Retrieved evidence:\n{evidence_text}\n\n"
                f"Answer: {answer}"
            ),
            "max_output_tokens": max_output_tokens,
        }
        response = client.responses.create(**request_payload)
        text = self._extract_response_text(response)
        if not text.strip():
            logger.warning("llm.evaluate.empty_response provider=openai model=%s", model)
            fallback = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a RAG answer judge. Evaluate the answer against the "
                            "question and the retrieved evidence for relevance, grounding, "
                            "and hallucination risk. Return a JSON object with exactly these "
                            'keys: {"verdict":"match or mismatch","reasoning":"one short explanation"}'
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Question: {query}\n"
                            f"Retrieval query: {retrieval_query or 'n/a'}\n\n"
                            f"Retrieved evidence:\n{evidence_text}\n\n"
                            f"Answer: {answer}"
                        ),
                    },
                ],
            )
            text = (fallback.choices[0].message.content or "").strip()
        logger.info("llm.evaluate.complete provider=openai model=%s", model)

        verdict = "unknown"
        reasoning = text or "Judge model returned an empty response."
        if text:
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
            try:
                payload = json.loads(cleaned)
                verdict = str(payload.get("verdict", verdict)).strip().lower() or verdict
                reasoning = str(payload.get("reasoning", reasoning)).strip() or reasoning
                return {"verdict": verdict, "reasoning": reasoning}
            except json.JSONDecodeError:
                pass

        for line in text.splitlines():
            if line.upper().startswith("VERDICT:"):
                verdict = line.split(":", 1)[1].strip().lower()
            if line.upper().startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()
        return {"verdict": verdict, "reasoning": reasoning}
