import argparse
import json
import sys

from graph import run_investigation

sys.stdout.reconfigure(encoding="utf-8")


def pretty_answer(answer: dict) -> str:
    lines = []
    lines.append("EXECUTIVE SUMMARY")
    lines.append("  " + (answer.get("executive_summary") or ""))
    lines.append("")
    lines.append("MAIN CAUSE")
    lines.append("  " + (answer.get("main_cause") or ""))
    lines.append("")
    lines.append("KEY DRIVERS")
    for d in answer.get("key_drivers") or []:
        lines.append("  - " + d)
    lines.append("")
    lines.append("EVIDENCE")
    for e in answer.get("evidence") or []:
        lines.append("  - " + e)
    lines.append("")
    lines.append("RECOMMENDED INVESTIGATION")
    for r in answer.get("recommended_investigation") or []:
        lines.append("  - " + r)
    if answer.get("chart_captions"):
        lines.append("")
        lines.append("CHARTS")
        for cap in answer.get("chart_captions") or []:
            lines.append("  - " + cap)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LangGraph sales-analyst investigation.")
    parser.add_argument("question", nargs="?", default="Why did revenue fall in August 2026?")
    parser.add_argument("--json", action="store_true", help="Output raw JSON state instead of a formatted answer.")
    args = parser.parse_args()

    state = run_investigation(args.question)

    if args.json:
        print(json.dumps(state, indent=2, default=str))
        return

    print("QUESTION:", args.question)
    print()
    print("INVESTIGATION PLAN")
    for i, step in enumerate(state.get("plan", []), 1):
        print(f"  {i}. {step}")
    print()
    print("QUERIES EXECUTED")
    for q in state.get("queries", []):
        print("  -", q.replace("\n", " "))
    print()
    print("OBSERVATIONS")
    for obs in state.get("observations", []):
        print("  " + obs.replace("\n", "\n  "))
        print("  ---")
    print()
    print("FINAL ANSWER")
    print(pretty_answer(state.get("final_answer", {})))


if __name__ == "__main__":
    main()
