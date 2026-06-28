import csv
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .reasoning import build_reasoning
from .scoring import (
    compute_availability_score,
    compute_behavior_score,
    compute_experience_score,
    compute_final_score,
    compute_location_score,
    compute_skill_score,
    extract_skill_flags,
)


MAX_NOTICE_PERIOD = 120


def extract_candidate_record(raw_candidate):
    profile = raw_candidate.get("profile", {}) if isinstance(raw_candidate, dict) else {}
    signals = raw_candidate.get("redrob_signals", {}) if isinstance(raw_candidate, dict) else {}
    skills = raw_candidate.get("skills", []) if isinstance(raw_candidate, dict) else []

    candidate_id = raw_candidate.get("candidate_id", "")
    years_of_experience = profile.get("years_of_experience", 0.0)
    current_title = profile.get("current_title", "")
    country = profile.get("country", "")
    matched_skills = sorted(extract_skill_flags(skills))

    recruiter_response_rate = signals.get("recruiter_response_rate", 0.0)
    interview_completion_rate = signals.get("interview_completion_rate", 0.0)
    github_activity_score = signals.get("github_activity_score", 0.0)
    saved_by_recruiters_30d = signals.get("saved_by_recruiters_30d", 0)
    open_to_work_flag = signals.get("open_to_work_flag", False)
    notice_period_days = signals.get("notice_period_days", MAX_NOTICE_PERIOD)

    skill_score = compute_skill_score(matched_skills)
    experience_score = compute_experience_score(years_of_experience)
    behavior_score = compute_behavior_score(
        recruiter_response_rate,
        interview_completion_rate,
        github_activity_score,
        saved_by_recruiters_30d,
    )
    availability_score = compute_availability_score(open_to_work_flag, notice_period_days)
    location_score = compute_location_score(country)
    final_score = compute_final_score(
        skill_score,
        experience_score,
        behavior_score,
        availability_score,
        location_score,
    )

    record = {
        "candidate_id": candidate_id,
        "years_of_experience": float(years_of_experience or 0.0),
        "current_title": current_title,
        "country": country,
        "matched_skills": matched_skills,
        "recruiter_response_rate": float(recruiter_response_rate or 0.0),
        "interview_completion_rate": float(interview_completion_rate or 0.0),
        "github_activity_score": float(github_activity_score or 0.0),
        "saved_by_recruiters_30d": float(saved_by_recruiters_30d or 0.0),
        "open_to_work_flag": bool(open_to_work_flag),
        "notice_period_days": int(notice_period_days or 0),
        "skill_score": skill_score,
        "experience_score": experience_score,
        "behavior_score": behavior_score,
        "availability_score": availability_score,
        "location_score": location_score,
        "score": final_score,
    }
    record["reasoning"] = build_reasoning(record)
    return record


def load_candidates(path):
    path = Path(path)
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading candidates", unit="candidate"):
            line = line.strip()
            if not line:
                continue
            raw_candidate = json.loads(line)
            records.append(extract_candidate_record(raw_candidate))
    return pd.DataFrame(records)


def write_submission(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for _, row in df.iterrows():
            writer.writerow([
                row["candidate_id"],
                int(row["rank"]),
                f"{row["score"]:.4f}",
                row["reasoning"],
            ])
