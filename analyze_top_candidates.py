import argparse
import csv
import json
from pathlib import Path

import pandas as pd

AI_TITLE_KEYWORDS = [
    "ml",
    "ai",
    "machine learning",
    "deep learning",
    "data scientist",
    "data science",
    "machine learning engineer",
    "ml engineer",
    "data engineer",
    "nlp",
    "computer vision",
    "research scientist",
    "mlops",
    "ai engineer",
    "ai specialist",
    "ai research",
    "search engineer",
]

NON_AI_TITLE_KEYWORDS = [
    "marketing",
    "hr",
    "human resources",
    "accountant",
    "customer support",
    "support specialist",
    "operations manager",
    "content writer",
    "sales",
    "recruiter",
]

AI_SKILL_KEYWORDS = [
    "python",
    "nlp",
    "llm",
    "fine-tuning",
    "lora",
    "qlora",
    "peft",
    "embeddings",
    "retrieval",
    "rag",
    "milvus",
    "pinecone",
    "qdrant",
    "faiss",
    "weaviate",
    "elasticsearch",
    "opensearch",
    "sentence transformers",
    "recommendation",
    "ranking",
    "vector database",
]

LOW_GITHUB_THRESHOLD = 25.0
EXPERIENCE_THRESHOLD_YEARS = 3.0


def normalize_text(value):
    if value is None:
        return ""
    return str(value).strip().lower()


def has_any_keyword(text, keywords):
    text = normalize_text(text)
    return any(keyword in text for keyword in keywords)


def extract_top_skills(candidate):
    skills = candidate.get("skills", [])
    if not isinstance(skills, list) or not skills:
        return []
    sorted_skills = sorted(
        [skill for skill in skills if isinstance(skill, dict)],
        key=lambda s: (-int(s.get("endorsements", 0)), s.get("name", "")),
    )
    names = [normalize_text(skill.get("name", "")) for skill in sorted_skills if normalize_text(skill.get("name", ""))]
    ai_skills = []
    for name in names:
        if any(keyword in name for keyword in AI_SKILL_KEYWORDS):
            ai_skills.append(name)
    return ai_skills[:5] if ai_skills else names[:5]


def load_submission(path):
    path = Path(path)
    return pd.read_csv(path)


def load_candidates(path, candidate_ids=None):
    path = Path(path)
    selected = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            candidate = json.loads(line)
            cid = candidate.get("candidate_id")
            if candidate_ids is None or cid in candidate_ids:
                selected[cid] = candidate
                if candidate_ids is not None and len(selected) >= len(candidate_ids):
                    break
    return selected


def build_suspicious_flags(candidate, top_skills):
    flags = []
    current_title = normalize_text(candidate.get("profile", {}).get("current_title", ""))
    if has_any_keyword(current_title, NON_AI_TITLE_KEYWORDS):
        flags.append("marketing/HR/accountant/sales/recruiter title")

    github_score = candidate.get("redrob_signals", {}).get("github_activity_score")
    if github_score is None or float(github_score or 0.0) < LOW_GITHUB_THRESHOLD:
        flags.append("low github activity")

    if not top_skills:
        flags.append("no AI skills")

    years = float(candidate.get("profile", {}).get("years_of_experience", 0.0) or 0.0)
    if years < EXPERIENCE_THRESHOLD_YEARS:
        flags.append("experience < 3 years")

    return "; ".join(flags) if flags else ""


def is_ai_title(title):
    return has_any_keyword(title, AI_TITLE_KEYWORDS)


def build_report_row(candidate, submission_row):
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    current_title = profile.get("current_title", "")
    years_of_experience = float(profile.get("years_of_experience", 0.0) or 0.0)
    country = profile.get("country", "")
    top_skills = extract_top_skills(candidate)
    final_score = float(submission_row.get("score", 0.0))
    suspicious = build_suspicious_flags(candidate, top_skills)

    return {
        "candidate_id": submission_row["candidate_id"],
        "current_title": current_title,
        "years_of_experience": years_of_experience,
        "country": country,
        "top_skills": ", ".join(top_skills),
        "final_score": final_score,
        "suspicious_flags": suspicious,
    }


def compute_statistics(rows):
    ai_titles = 0
    non_ai_titles = 0
    total_experience = 0.0
    india_count = 0
    open_to_work_count = 0
    total_github = 0.0
    count = len(rows)

    for row in rows:
        current_title = normalize_text(row["current_title"])
        if is_ai_title(current_title):
            ai_titles += 1
        else:
            non_ai_titles += 1

        total_experience += row["years_of_experience"]
        if normalize_text(row["country"]) == "india":
            india_count += 1

        candidate = row.get("candidate_obj")
        open_to_work = candidate.get("redrob_signals", {}).get("open_to_work_flag", False)
        github_score = candidate.get("redrob_signals", {}).get("github_activity_score", 0.0)
        if bool(open_to_work):
            open_to_work_count += 1
        total_github += float(github_score or 0.0)

    stats = {
        "AI/ML titles count": ai_titles,
        "Non-AI titles count": non_ai_titles,
        "Average experience": round(total_experience / count, 2) if count else 0.0,
        "India candidates %": round(100.0 * india_count / count, 2) if count else 0.0,
        "Open to work %": round(100.0 * open_to_work_count / count, 2) if count else 0.0,
        "Average github activity score": round(total_github / count, 2) if count else 0.0,
    }
    return stats


def save_analysis(rows, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "candidate_id",
        "current_title",
        "years_of_experience",
        "country",
        "top_skills",
        "final_score",
        "suspicious_flags",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})


def analyze_top_candidates(submission_path, candidates_path, output_path):
    submission = load_submission(submission_path)
    top20 = submission.head(20)
    candidate_ids = set(top20["candidate_id"].astype(str).tolist())
    candidates = load_candidates(candidates_path, candidate_ids)

    report_rows = []
    for _, submission_row in top20.iterrows():
        candidate_id = submission_row["candidate_id"]
        candidate = candidates.get(candidate_id, {})
        row = build_report_row(candidate, submission_row)
        row["candidate_obj"] = candidate
        report_rows.append(row)

    stats = compute_statistics(report_rows)

    print("Top 20 candidate analysis")
    print(submission.head(20)[["candidate_id", "score"]].to_string(index=False))
    print()
    print("Statistics")
    for key, value in stats.items():
        print(f"- {key}: {value}")

    print()
    suspicious = [row for row in report_rows if row["suspicious_flags"]]
    print(f"Suspicious candidates: {len(suspicious)}")
    for row in suspicious:
        print(f"- {row['candidate_id']}: {row['suspicious_flags']}")

    # Remove internal object before saving
    for row in report_rows:
        row.pop("candidate_obj", None)

    save_analysis(report_rows, output_path)
    print(f"Saved top 20 analysis to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze the top candidates from submission.csv.")
    parser.add_argument(
        "--submission",
        default="submission.csv",
        help="Path to submission CSV",
    )
    parser.add_argument(
        "--candidates",
        default="data/candidates.jsonl",
        help="Path to candidates.jsonl",
    )
    parser.add_argument(
        "--output",
        default="top20_analysis.csv",
        help="Path to output analysis CSV",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    analyze_top_candidates(args.submission, args.candidates, args.output)


if __name__ == "__main__":
    main()
