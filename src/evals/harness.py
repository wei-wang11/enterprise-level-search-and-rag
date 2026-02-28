import json
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def score_case(case: dict, output) -> dict:
    return {
        "query": case["query"],
        "usecase": case.get("usecase"),
        "answered": bool(output.answer),
        "citation_count": len(output.citations),
    }


def write_report(usecase: str, results: list[dict]) -> None:
    report_path = Path("reports")
    report_path.mkdir(exist_ok=True)
    with (report_path / f"{usecase}.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


def summarize(results: list[dict]) -> dict:
    total = len(results)
    answered = sum(1 for result in results if result["answered"])
    return {
        "total": total,
        "answered": answered,
        "answer_rate": answered / total if total else 0.0,
    }


def run_eval(usecase: str, dataset_path: str, pipeline):
    cases = load_jsonl(str(Path(dataset_path) / "queries.jsonl"))
    results = []
    for case in cases:
        out = pipeline.run(user=case["user"], query=case["query"], usecase=usecase)
        results.append(score_case(case, out))
    write_report(usecase, results)
    return summarize(results)
