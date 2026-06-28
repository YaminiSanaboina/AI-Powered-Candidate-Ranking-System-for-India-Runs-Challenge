import argparse

from .io import load_candidates, write_submission


def parse_args():
    parser = argparse.ArgumentParser(description="Redrob candidate ranking CLI")
    parser.add_argument(
        "--input",
        default="data/candidates.jsonl",
        help="Path to the input candidates.jsonl file",
    )
    parser.add_argument(
        "--output",
        default="submission.csv",
        help="Path to the output submission CSV file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    candidates = load_candidates(args.input)
    candidates.sort_values(by=["score", "candidate_id"], ascending=[False, True], inplace=True)
    candidates = candidates.head(100).reset_index(drop=True)
    candidates["rank"] = candidates.index + 1
    submission = candidates[["candidate_id", "rank", "score", "reasoning"]]
    write_submission(submission, args.output)
    print(f"Generated submission with {len(submission)} rows at {args.output}")
