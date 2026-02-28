from pathlib import Path


class PromptLoader:
    def __init__(self, root: str = "src/rag/prompts") -> None:
        self.root = Path(root)

    def load(self, prompt_name: str) -> str:
        prompt_path = self.root / prompt_name
        with prompt_path.open("r", encoding="utf-8") as handle:
            return handle.read()
