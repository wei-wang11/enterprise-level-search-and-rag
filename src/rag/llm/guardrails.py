def apply_guardrails(raw: str, hits: list, cfg) -> str:
    generation_cfg = getattr(cfg, "generation", None)

    if generation_cfg and getattr(generation_cfg, "refusal_when_low_evidence", False) and not hits:
        return "I do not know based on the available evidence. Please provide more specific material."

    if cfg.security.redact_secrets:
        return raw.replace("OPENAI_API_KEY", "[REDACTED]")

    return raw
