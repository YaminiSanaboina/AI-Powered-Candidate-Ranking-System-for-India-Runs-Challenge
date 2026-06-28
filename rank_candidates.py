import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer, util
    SEMANTIC_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    SentenceTransformer = None
    util = None
    SEMANTIC_AVAILABLE = False

try:
    import torch
except ImportError:
    torch = None

EMBEDDING_CACHE_DIR = Path("cache")
SEMANTIC_MODEL_NAME = "all-MiniLM-L6-v2"
SEMANTIC_BATCH_SIZE = 128

SKILL_KEYWORDS = {
    "Python": ["python"],
    "NLP": ["nlp", "natural language processing"],
    "LLM": ["llm", "large language model", "large language models"],
    "Fine-tuning LLMs": ["fine-tuning llms", "fine tuning llms", "fine-tuning llm", "fine tuning llm", "fine tuning"],
    "LoRA": ["lora"],
    "QLoRA": ["qlora"],
    "PEFT": ["peft"],
    "Embeddings": ["embeddings", "vector embeddings"],
    "Retrieval": ["retrieval", "retrieval-augmented", "retrieval augmented"],
    "RAG": ["rag", "retrieval augmented generation"],
    "Milvus": ["milvus"],
    "Pinecone": ["pinecone"],
    "Qdrant": ["qdrant"],
    "FAISS": ["faiss", "facebook ai similarity search"],
    "Weaviate": ["weaviate"],
    "Vector Database": ["vector database", "vector databases", "vector db"],
}

BOOST_KEYWORDS = [
    "embeddings",
    "retrieval",
    "ranking systems",
    "vector database",
    "pinecone",
    "qdrant",
    "weaviate",
    "milvus",
    "faiss",
    "elasticsearch",
    "opensearch",
    "sentence transformers",
    "rag",
    "llm",
    "fine tuning",
    "fine-tuning",
    "lora",
    "qlora",
    "peft",
    "ndcg",
    "mrr",
    "map",
    "ab testing",
    "recommendation systems",
]

EDUCATION_KEYWORDS = [
    "bachelor",
    "master",
    "mba",
    "phd",
    "degree",
    "university",
    "college",
    "graduat",
    "diploma",
]

CERTIFICATION_KEYWORDS = [
    "certified",
    "certification",
    "aws certified",
    "azure certified",
    "google certified",
    "professional certificate",
    "pmp",
    "scrum",
    "keras",
]

PROJECT_KEYWORDS = [
    "project",
    "launched",
    "deployed",
    "production",
    "prototype",
    "proof of concept",
    "poc",
    "built",
    "designed",
]

PRODUCTION_KEYWORDS = [
    "retrieval systems",
    "ranking systems",
    "vector databases",
    "embeddings",
    "search infrastructure",
    "production ml",
    "production system",
    "production ml",
    "deployed",
    "deployed in production",
    "search infrastructure",
    "search infra",
    "ranking system",
    "retrieval system",
    "real-time",
    "online serving",
    "inference pipeline",
    "serving model",
    "productionized",
    "production data",
    "production deployment",
    "deployment",
    "ndcg",
    "map",
    "mrr",
    "ab testing",
]

TITLE_BOOST_KEYWORDS = [
    "ai engineer",
    "ml engineer",
    "machine learning engineer",
    "applied scientist",
    "nlp engineer",
    "data scientist",
    "search engineer",
    "recommendation engineer",
    "retrieval engineer",
]

BAD_TITLE_KEYWORDS = [
    "marketing manager",
    "hr manager",
    "recruiter",
    "accountant",
    "operations manager",
    "customer support",
    "content writer",
    "sales",
]

SCORE_WEIGHTS = {
    "semantic": 0.20,
    "behavior": 0.22,
    "experience": 0.18,
    "production": 0.10,
    "availability": 0.08,
    "skill": 0.12,
    "relevance": 0.10,
}

MAX_BEHAVIOR_SAVED = 20
MAX_NOTICE_PERIOD = 120


def normalize_text(value):
    if value is None:
        return ""
    return str(value).strip().lower()


def load_job_description(path):
    path = Path(path)
    if path.suffix.lower() == ".docx":
        document = Document(path)
        paragraphs = [p.text for p in document.paragraphs if p.text]
        return " ".join(paragraphs)
    return path.read_text(encoding="utf-8")


def get_job_embedding_cache_path(job_description_path, model_name):
    EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_model_name = re.sub(r"[^a-z0-9_]+", "_", model_name.lower())
    return EMBEDDING_CACHE_DIR / f"{job_description_path.stem}_{safe_model_name}.npy"


def load_cached_job_embedding(job_description_path, model):
    job_description_path = Path(job_description_path)
    cache_path = get_job_embedding_cache_path(job_description_path, SEMANTIC_MODEL_NAME)
    if cache_path.exists():
        try:
            data = np.load(cache_path)
            if torch is not None:
                return torch.from_numpy(data)
            return data
        except Exception:
            pass

    job_text = load_job_description(job_description_path)
    embedding = model.encode(job_text, convert_to_tensor=True)
    try:
        np.save(cache_path, embedding.cpu().numpy())
    except Exception:
        pass
    return embedding


def create_candidate_text(raw_candidate):
    profile = raw_candidate.get("profile", {}) if isinstance(raw_candidate, dict) else {}
    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_title", ""),
    ]
    skills = raw_candidate.get("skills", []) if isinstance(raw_candidate, dict) else []
    if isinstance(skills, list):
        parts.append(" ".join(normalize_text(skill.get("name", "")) for skill in skills if isinstance(skill, dict)))
    career_history = raw_candidate.get("career_history", []) if isinstance(raw_candidate, dict) else []
    for item in career_history:
        if not isinstance(item, dict):
            continue
        parts.append(item.get("title", ""))
        parts.append(item.get("description", ""))
    return " ".join([normalize_text(part) for part in parts if part])


def count_keyword_hits(text, keywords):
    text = normalize_text(text)
    if not text:
        return 0
    return sum(1 for keyword in keywords if keyword in text)


def any_keyword_in_text(text, keywords):
    text = normalize_text(text)
    return any(keyword in text for keyword in keywords)


def extract_profile_texts(raw_candidate):
    profile = raw_candidate.get("profile", {}) if isinstance(raw_candidate, dict) else {}
    headline = profile.get("headline", "")
    summary = profile.get("summary", "")
    current_title = profile.get("current_title", "")
    career_items = raw_candidate.get("career_history", []) if isinstance(raw_candidate, dict) else []
    career_titles = []
    career_descriptions = []
    for item in career_items:
        if not isinstance(item, dict):
            continue
        career_titles.append(item.get("title", ""))
        career_descriptions.append(item.get("description", ""))
    return {
        "headline": headline,
        "summary": summary,
        "current_title": current_title,
        "career_titles": career_titles,
        "career_descriptions": career_descriptions,
    }


def extract_skill_flags(skill_entries):
    if not isinstance(skill_entries, list):
        return set()

    normalized_skills = [normalize_text(skill.get("name", "")) for skill in skill_entries if isinstance(skill, dict)]
    matched = set()
    for canonical, keywords in SKILL_KEYWORDS.items():
        for keyword in keywords:
            if any(keyword in skill_text for skill_text in normalized_skills):
                matched.add(canonical)
                break
    return matched


def compute_skill_score(matched_skills, text_skill_hits):
    matched_ratio = len(matched_skills) / len(SKILL_KEYWORDS)
    text_ratio = min(1.0, text_skill_hits / 7.0)
    return 100.0 * (0.65 * matched_ratio + 0.35 * text_ratio)


def compute_experience_score(years_of_experience):
    years = np.clip(float(years_of_experience or 0.0), 0.0, 50.0)
    if years < 2.0:
        return 10.0 * (years / 2.0)
    if years < 3.0:
        return 25.0 + 25.0 * (years - 2.0)
    if years < 4.0:
        return 60.0 + 25.0 * (years - 3.0)
    if years <= 5.0:
        return 85.0 + 15.0 * (years - 4.0)
    if years <= 9.0:
        return 100.0
    if years <= 11.0:
        return 90.0
    if years <= 15.0:
        return 75.0
    return 60.0


def compute_behavior_score(recruiter_response_rate, interview_completion_rate, github_activity_score, saved_by_recruiters_30d, open_to_work_flag):
    response = np.clip(float(recruiter_response_rate or 0.0), 0.0, 1.0)
    interview = np.clip(float(interview_completion_rate or 0.0), 0.0, 1.0)
    github = np.clip(float(github_activity_score or 0.0), 0.0, 100.0) / 100.0
    saved = np.clip(float(saved_by_recruiters_30d or 0.0), 0.0, MAX_BEHAVIOR_SAVED) / MAX_BEHAVIOR_SAVED
    open_flag = 1.0 if bool(open_to_work_flag) else 0.4
    return 100.0 * np.average([response, interview, github, saved, open_flag], weights=[0.24, 0.24, 0.20, 0.20, 0.12])


def compute_availability_score(notice_period_days):
    notice = np.clip(float(notice_period_days if notice_period_days is not None else MAX_NOTICE_PERIOD), 0.0, MAX_NOTICE_PERIOD)
    return 100.0 * (1.0 - notice / MAX_NOTICE_PERIOD)


def compute_semantic_score(candidate_embedding, jd_embedding):
    if candidate_embedding is None:
        return 0.0
    similarity = util.cos_sim(candidate_embedding, jd_embedding).item()
    return float(np.clip((similarity + 1.0) / 2.0 * 100.0, 0.0, 100.0))


def compute_semantic_scores(df, jd_embedding, model):
    if not SEMANTIC_AVAILABLE or model is None or util is None or jd_embedding is None:
        df["semantic_score"] = 0.0
        return df

    candidate_texts = df["candidate_text"].fillna("").tolist()
    all_scores = []
    for start in range(0, len(candidate_texts), SEMANTIC_BATCH_SIZE):
        batch_texts = candidate_texts[start : start + SEMANTIC_BATCH_SIZE]
        try:
            batch_embeddings = model.encode(batch_texts, batch_size=SEMANTIC_BATCH_SIZE, convert_to_tensor=True)
            batch_scores = util.cos_sim(batch_embeddings, jd_embedding).cpu().numpy().reshape(-1)
        except Exception:
            batch_scores = np.zeros(len(batch_texts), dtype=float)
        all_scores.extend(batch_scores)
    df["semantic_score"] = [float(np.clip((score + 1.0) / 2.0 * 100.0, 0.0, 100.0)) for score in all_scores]
    return df


def compute_final_score(row):
    weighted_score = (
        SCORE_WEIGHTS["semantic"] * float(row.get("semantic_score", 0.0))
        + SCORE_WEIGHTS["experience"] * float(row.get("experience_score", 0.0))
        + SCORE_WEIGHTS["behavior"] * float(row.get("behavior_score", 0.0))
        + SCORE_WEIGHTS["availability"] * float(row.get("availability_score", 0.0))
        + SCORE_WEIGHTS["production"] * float(row.get("production_score", 0.0))
        + SCORE_WEIGHTS["skill"] * float(row.get("skill_score", 0.0))
        + SCORE_WEIGHTS["relevance"] * float(row.get("relevance_score", 0.0))
    )
    title_boost = float(row.get("title_boost", 0.0))
    ai_evidence_boost = float(row.get("ai_evidence_boost", 0.0))
    title_penalty = float(row.get("title_penalty", 0.0))
    anti_penalty = float(row.get("anti_penalty", 0.0))
    return float(np.clip(weighted_score + title_boost + ai_evidence_boost - title_penalty - anti_penalty, 0.0, 100.0))


def compute_production_score(all_text):
    production_hits = count_keyword_hits(all_text, PRODUCTION_KEYWORDS)
    if production_hits >= 6:
        return 100.0
    return 100.0 * (production_hits / 6.0)


def compute_location_score(country):
    return 100.0 if normalize_text(country) == "india" else 0.0


def compute_title_boost(current_title):
    title_lower = normalize_text(current_title)
    return 8.0 if any(keyword in title_lower for keyword in TITLE_BOOST_KEYWORDS) else 0.0


def compute_title_penalty(current_title, career_titles, all_text):
    title_text = " ".join([current_title] + career_titles)
    bad_matches = count_keyword_hits(title_text, BAD_TITLE_KEYWORDS)
    if bad_matches == 0:
        return 0.0
    if any_keyword_in_text(all_text, BOOST_KEYWORDS + list(SKILL_KEYWORDS.keys())):
        return 4.0
    return 10.0


def has_ai_evidence(text, career_descriptions):
    text = normalize_text(text)
    career_text = " ".join(normalize_text(desc) for desc in career_descriptions)
    return any_keyword_in_text(text, BOOST_KEYWORDS + ["ai", "ml", "machine learning", "deep learning", "model", "search", "retrieval"]) or any_keyword_in_text(career_text, BOOST_KEYWORDS + ["ai", "ml", "machine learning", "deep learning", "model", "search", "retrieval"])


def compute_anti_honeypot_penalty(matched_skills, all_text, current_title, career_titles, career_descriptions, years_of_experience, summary_text):
    penalties = 0.0
    ai_text_count = count_keyword_hits(all_text, BOOST_KEYWORDS + [keyword for keywords in SKILL_KEYWORDS.values() for keyword in keywords])
    has_ai_title = any_keyword_in_text(current_title, BOOST_KEYWORDS + ["ml", "ai", "machine learning", "deep learning"])
    has_ai_history = any(any_keyword_in_text(text, BOOST_KEYWORDS + ["model", "ml", "ai", "search", "retrieval"]) for text in career_titles + career_descriptions)

    if len(matched_skills) >= 6 and ai_text_count < 2:
        penalties += 12.0
    if len(matched_skills) >= 4 and not has_ai_title and not has_ai_history:
        penalties += 8.0
    if len(matched_skills) >= 5 and years_of_experience <= 2 and not has_ai_history:
        penalties += 10.0
    if any(phrase in summary_text for phrase in ["transitioning", "looking to move", "pivot", "switch into", "interested in transitioning"]):
        penalties += 4.0
    if years_of_experience > 0:
        career_months = 0
        raw_career = career_descriptions
        if career_descriptions:
            career_months = sum(0 for _ in career_descriptions)
        if years_of_experience < 1.0 and len(matched_skills) >= 5:
            penalties += 4.0
    return min(20.0, penalties)


def build_reasoning(candidate_id, years_of_experience, current_title, matched_skills, production_score, recruiter_response_rate, interview_completion_rate, github_activity_score, saved_by_recruiters_30d, open_to_work_flag, notice_period_days, semantic_score=None, final_score=None):
    strengths = []
    weaknesses = []
    signals = []

    skill_count = len(matched_skills) if matched_skills else 0
    
    # Skills
    if matched_skills:
        top_skills = sorted(
            matched_skills,
            key=lambda skill: list(SKILL_KEYWORDS.keys()).index(skill) if skill in SKILL_KEYWORDS else len(SKILL_KEYWORDS),
        )[:4]
        if skill_count >= 8:
            strengths.append(f"{skill_count} core AI skills")
        elif skill_count >= 5:
            strengths.append(f"{skill_count} strong skills: {', '.join(top_skills[:3])}")
        elif skill_count >= 3:
            strengths.append(f"Skills: {', '.join(top_skills[:3])}")
        else:
            strengths.append(f"Key skills: {', '.join(top_skills)}")
    else:
        weaknesses.append("No AI skills detected")

    # Experience
    if years_of_experience >= 10:
        strengths.append(f"{years_of_experience:.0f}+ years experience")
    elif years_of_experience >= 7:
        strengths.append(f"{years_of_experience:.1f}y ML/AI")
    elif years_of_experience >= 5:
        strengths.append(f"{years_of_experience:.1f}y professional")
    elif years_of_experience >= 3:
        signals.append(f"{years_of_experience:.1f}y career")
    else:
        weaknesses.append(f"Early stage: {years_of_experience:.1f}y")

    # Production
    if production_score >= 80:
        strengths.append("Deployed production systems")
    elif production_score >= 60:
        strengths.append("Production ML exp")
    elif production_score >= 40:
        signals.append("Some production work")
    else:
        weaknesses.append("Limited production")

    # Semantic
    if semantic_score is not None and semantic_score > 0:
        if semantic_score >= 80:
            strengths.append("Excellent fit")
        elif semantic_score >= 65:
            strengths.append("Strong alignment")
        elif semantic_score >= 50:
            signals.append("Moderate relevance")
        else:
            weaknesses.append("Limited relevance")

    # Behavior: Recruiter
    if recruiter_response_rate >= 0.75:
        signals.append(f"High recruiter response")
    elif recruiter_response_rate >= 0.50:
        signals.append(f"Good engagement")
    elif recruiter_response_rate > 0:
        weaknesses.append(f"Low responses")

    # Behavior: Interviews
    if interview_completion_rate >= 0.75:
        signals.append("Strong interview progression")
    elif interview_completion_rate >= 0.50:
        signals.append("Regular interviews")
    elif interview_completion_rate > 0:
        weaknesses.append("Interview gaps")

    # Behavior: GitHub
    if github_activity_score >= 75:
        signals.append("Active contributor")
    elif github_activity_score >= 50:
        signals.append("GitHub presence")
    elif github_activity_score > 0:
        weaknesses.append("Limited GitHub activity")
    else:
        weaknesses.append("No GitHub profile")

    # Behavior: Recruiter saves
    if saved_by_recruiters_30d >= 4:
        signals.append("High recruiter saves")
    elif saved_by_recruiters_30d >= 2:
        signals.append("Recruiter interest")

    # Availability
    avail = []
    if open_to_work_flag:
        avail.append("Actively seeking")
    else:
        avail.append("Passive candidate")
    
    if notice_period_days is not None:
        if notice_period_days <= 7:
            avail.append("Immediate start")
        elif notice_period_days <= 30:
            avail.append(f"{int(notice_period_days)}d notice")
        else:
            avail.append("Extended notice")

    # Recommendation
    if final_score is not None:
        if final_score >= 75:
            rec = "Highly Recommended"
        elif final_score >= 70:
            rec = "Recommended"
        elif final_score >= 65:
            rec = "Good Fit"
        else:
            rec = "Consider"
    else:
        rec = "Review"

    # Risk
    risk_count = 0
    if recruiter_response_rate <= 0.1:
        risk_count += 1
    if interview_completion_rate <= 0.1:
        risk_count += 1
    if github_activity_score < 15 and skill_count >= 5:
        risk_count += 1
    if years_of_experience < 2 and skill_count > 5:
        risk_count += 1
    if production_score < 20 and skill_count >= 6:
        risk_count += 1

    if risk_count >= 4:
        risk = "High"
    elif risk_count >= 2:
        risk = "Medium"
    else:
        risk = "Low"

    # Output
    out = ""
    if strengths:
        out += "Strengths\\n" + "\\n".join(f"✓ {s}" for s in strengths[:4]) + "\\n"
    if weaknesses:
        out += "Weaknesses\\n" + "\\n".join(f"• {w}" for w in weaknesses[:3]) + "\\n"
    if signals:
        out += "Signals\\n" + "\\n".join(f"• {s}" for s in signals[:2]) + "\\n"
    out += f"Availability: {', '.join(avail)}\\n"
    out += f"\\nRecommendation: {rec}\\nRisk: {risk}"
    
    return out


def extract_candidate_record(raw_candidate):
    profile = raw_candidate.get("profile", {}) if isinstance(raw_candidate, dict) else {}
    signals = raw_candidate.get("redrob_signals", {}) if isinstance(raw_candidate, dict) else {}
    skills = raw_candidate.get("skills", []) if isinstance(raw_candidate, dict) else []

    candidate_id = raw_candidate.get("candidate_id", "")
    years_of_experience = profile.get("years_of_experience", 0.0)
    current_title = profile.get("current_title", "")
    country = profile.get("country", "")

    text_fields = extract_profile_texts(raw_candidate)
    headline_text = normalize_text(text_fields["headline"])
    summary_text = normalize_text(text_fields["summary"])
    title_text = normalize_text(text_fields["current_title"])
    career_titles = [normalize_text(t) for t in text_fields["career_titles"]]
    career_descriptions = [normalize_text(d) for d in text_fields["career_descriptions"]]
    all_text = " ".join([headline_text, summary_text, title_text] + career_titles + career_descriptions)

    matched_skills = extract_skill_flags(skills)
    text_skill_hits = count_keyword_hits(all_text, BOOST_KEYWORDS)

    recruiter_response_rate = signals.get("recruiter_response_rate", 0.0)
    interview_completion_rate = signals.get("interview_completion_rate", 0.0)
    github_activity_score = signals.get("github_activity_score", 0.0)
    saved_by_recruiters_30d = signals.get("saved_by_recruiters_30d", 0)
    open_to_work_flag = signals.get("open_to_work_flag", False)
    notice_period_days = signals.get("notice_period_days", MAX_NOTICE_PERIOD)

    skill_score = compute_skill_score(matched_skills, text_skill_hits)
    experience_score = compute_experience_score(years_of_experience)
    behavior_score = compute_behavior_score(
        recruiter_response_rate,
        interview_completion_rate,
        github_activity_score,
        saved_by_recruiters_30d,
        open_to_work_flag,
    )
    availability_score = compute_availability_score(notice_period_days)
    production_score = compute_production_score(all_text)
    location_score = compute_location_score(country)

    title_boost = compute_title_boost(current_title)
    title_penalty = compute_title_penalty(current_title, career_titles, all_text)
    anti_penalty = compute_anti_honeypot_penalty(
        matched_skills,
        all_text,
        current_title,
        career_titles,
        career_descriptions,
        float(years_of_experience or 0.0),
        summary_text,
    )

    ai_evidence_boost = 0.0
    if not has_ai_evidence(summary_text, career_descriptions):
        anti_penalty += 5.0
    else:
        ai_evidence_boost = 4.0

    if float(years_of_experience or 0.0) < 2.0:
        anti_penalty += 15.0

    education_hits = count_keyword_hits(all_text, EDUCATION_KEYWORDS)
    certification_hits = count_keyword_hits(all_text, CERTIFICATION_KEYWORDS)
    project_hits = count_keyword_hits(all_text, PROJECT_KEYWORDS)
    education_score = 100.0 * min(1.0, education_hits / 2.0)
    certification_score = 100.0 * min(1.0, certification_hits / 2.0)
    project_score = 100.0 * min(1.0, project_hits / 4.0)
    relevance_score = float(np.clip(0.35 * education_score + 0.35 * certification_score + 0.30 * project_score, 0.0, 100.0))

    raw_score = (
        SCORE_WEIGHTS["semantic"] * 0.0
        + SCORE_WEIGHTS["experience"] * experience_score
        + SCORE_WEIGHTS["behavior"] * behavior_score
        + SCORE_WEIGHTS["availability"] * availability_score
        + SCORE_WEIGHTS["production"] * production_score
        + SCORE_WEIGHTS["skill"] * skill_score
        + SCORE_WEIGHTS["relevance"] * relevance_score
    )
    score = float(np.clip(raw_score + title_boost + ai_evidence_boost - title_penalty - anti_penalty, 0.0, 100.0))

    semantic_score = 0.0

    reasoning = build_reasoning(
        candidate_id=candidate_id,
        years_of_experience=float(years_of_experience or 0.0),
        current_title=current_title,
        matched_skills=matched_skills,
        production_score=production_score,
        recruiter_response_rate=float(recruiter_response_rate or 0.0),
        interview_completion_rate=float(interview_completion_rate or 0.0),
        github_activity_score=float(github_activity_score or 0.0),
        saved_by_recruiters_30d=float(saved_by_recruiters_30d or 0.0),
        open_to_work_flag=open_to_work_flag,
        notice_period_days=int(notice_period_days) if notice_period_days is not None else MAX_NOTICE_PERIOD,
        semantic_score=semantic_score,
        final_score=score,
    )

    title_consistency = 0.0
    if matched_skills and not has_ai_evidence(title_text, career_descriptions):
        title_consistency = -5.0

    return {
        "candidate_id": candidate_id,
        "years_of_experience": float(years_of_experience or 0.0),
        "current_title": current_title,
        "country": country,
        "matched_skills": sorted(matched_skills),
        "skill_score": skill_score,
        "experience_score": experience_score,
        "behavior_score": behavior_score,
        "availability_score": availability_score,
        "production_score": production_score,
        "relevance_score": relevance_score,
        "semantic_score": semantic_score,
        "title_boost": title_boost,
        "title_penalty": title_penalty,
        "anti_penalty": anti_penalty,
        "ai_evidence_boost": ai_evidence_boost,
        "candidate_text": create_candidate_text(raw_candidate),
        "title_consistency": title_consistency,
        "score": score,
        "reasoning": reasoning,
    }


def load_candidates(path):
    path = Path(path)
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading candidates", unit="candidates"):
            line = line.strip()
            if not line:
                continue
            candidate = json.loads(line)
            records.append(extract_candidate_record(candidate))
    return pd.DataFrame(records)


def rank_candidates(df, top_n=100):
    df = df.copy()
    df["score_rounded"] = df["score"].round(4)
    df.sort_values(by=["score_rounded", "candidate_id"], ascending=[False, True], inplace=True)
    df = df.drop(columns=["score_rounded"])
    df = df.reset_index(drop=True)
    df = df.head(top_n)
    df["rank"] = np.arange(1, len(df) + 1)
    df["score"] = df["score"].round(4)
    return df[["candidate_id", "rank", "score", "reasoning"]]


def title_distribution(df):
    titles = df["current_title"].fillna("").apply(normalize_text)
    ai_count = titles.apply(lambda t: any(keyword in t for keyword in TITLE_BOOST_KEYWORDS + ["data scientist", "ml", "ai", "machine learning", "search engineer", "recommendation"])).sum()
    non_ai_count = titles.apply(lambda t: any(keyword in t for keyword in BAD_TITLE_KEYWORDS)).sum()
    return ai_count, non_ai_count


def write_submission(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL, float_format="%.4f")


def parse_args():
    parser = argparse.ArgumentParser(description="Build a candidate ranking submission for the Redrob challenge.")
    parser.add_argument(
        "--input",
        default="data/candidates.jsonl",
        help="Path to candidates.jsonl",
    )
    parser.add_argument(
        "--output",
        default="submission.csv",
        help="Path to output submission CSV",
    )
    parser.add_argument(
        "--job-description",
        default="data/job_description.docx",
        help="Path to the job description file",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Number of top candidates to include in the submission",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = load_candidates(args.input)
    model = None
    jd_embedding = None

    if SEMANTIC_AVAILABLE:
        try:
            model = SentenceTransformer(SEMANTIC_MODEL_NAME)
            jd_embedding = load_cached_job_embedding(args.job_description, model)
        except Exception:
            model = None
            jd_embedding = None

    if jd_embedding is None:
        print("Warning: sentence-transformers not installed or cache unavailable. Semantic ranking disabled.")
        df["semantic_score"] = 0.0

    df = compute_semantic_scores(df, jd_embedding, model)
    df["score"] = df.apply(compute_final_score, axis=1)

    df["reasoning"] = df.apply(
        lambda row: build_reasoning(
            candidate_id=row["candidate_id"],
            years_of_experience=row["years_of_experience"],
            current_title=row["current_title"],
            matched_skills=row["matched_skills"],
            production_score=row["production_score"],
            recruiter_response_rate=row.get("recruiter_response_rate", 0.0),
            interview_completion_rate=row.get("interview_completion_rate", 0.0),
            github_activity_score=row.get("github_activity_score", 0.0),
            saved_by_recruiters_30d=row.get("saved_by_recruiters_30d", 0.0),
            open_to_work_flag=row.get("open_to_work_flag", False),
            notice_period_days=int(row.get("notice_period_days", MAX_NOTICE_PERIOD)),
            semantic_score=row.get("semantic_score", 0.0),
            final_score=row.get("score", 0.0),
        ),
        axis=1,
    )
    submission = rank_candidates(df, top_n=args.top_n)
    write_submission(submission, args.output)
    top_df = df.loc[df["candidate_id"].isin(submission["candidate_id"])].copy()
    ai_titles, non_ai_titles = title_distribution(top_df)
    min_exp = top_df["years_of_experience"].min()
    max_exp = top_df["years_of_experience"].max()
    print(f"Generated submission with {len(submission)} candidates at {args.output}")
    print(f"Title distribution in top {len(submission)}: AI/ML titles = {ai_titles}, Non-AI titles = {non_ai_titles}")
    print(f"Top 20 experience range: min = {min_exp:.1f}, max = {max_exp:.1f}")


if __name__ == "__main__":
    main()
