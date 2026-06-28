import numpy as np

SKILL_KEYWORDS = {
    "Python": ["python"],
    "NLP": ["nlp", "natural language processing"],
    "LLM": ["llm", "large language model", "large language models"],
    "Fine-tuning LLMs": ["fine-tuning llms", "fine tuning llms", "fine-tuning llm", "fine tuning llm", "fine tuning"],
    "LoRA": ["lora"],
    "Embeddings": ["embeddings", "vector embeddings"],
    "Retrieval": ["retrieval", "retrieval-augmented", "retrieval augmented"],
    "RAG": ["rag", "retrieval augmented generation"],
    "Milvus": ["milvus"],
    "Pinecone": ["pinecone"],
    "Qdrant": ["qdrant"],
    "FAISS": ["faiss", "facebook ai similarity search"],
    "Vector Database": ["vector database", "vector databases", "vector db"],
}

MAX_BEHAVIOR_SAVED = 20
MAX_NOTICE_PERIOD = 120


def normalize_text(value):
    if value is None:
        return ""
    return str(value).strip().lower()


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


def compute_skill_score(matched_skills):
    if not matched_skills:
        return 0.0
    return 100.0 * len(matched_skills) / len(SKILL_KEYWORDS)


def compute_experience_score(years_of_experience):
    years = np.clip(float(years_of_experience or 0.0), 0.0, 50.0)
    if years <= 5.0:
        return 100.0 * (years / 5.0)
    if years <= 9.0:
        return 100.0
    return max(0.0, 100.0 - (years - 9.0) * 6.25)


def compute_behavior_score(recruiter_response_rate, interview_completion_rate, github_activity_score, saved_by_recruiters_30d):
    response = np.clip(float(recruiter_response_rate or 0.0), 0.0, 1.0)
    interview = np.clip(float(interview_completion_rate or 0.0), 0.0, 1.0)
    github = np.clip(float(github_activity_score or 0.0), 0.0, 100.0) / 100.0
    saved = np.clip(float(saved_by_recruiters_30d or 0.0), 0.0, MAX_BEHAVIOR_SAVED) / MAX_BEHAVIOR_SAVED
    return 100.0 * np.mean([response, interview, github, saved])


def compute_availability_score(open_to_work_flag, notice_period_days):
    open_flag = 1.0 if bool(open_to_work_flag) else 0.25
    notice = np.clip(float(notice_period_days if notice_period_days is not None else MAX_NOTICE_PERIOD), 0.0, MAX_NOTICE_PERIOD)
    notice_score = 1.0 - (notice / MAX_NOTICE_PERIOD)
    return 100.0 * (0.55 * open_flag + 0.45 * notice_score)


def compute_location_score(country):
    return 100.0 if normalize_text(country) == "india" else 0.0


def compute_final_score(skill_score, experience_score, behavior_score, availability_score, location_score):
    score = (
        0.40 * skill_score
        + 0.25 * experience_score
        + 0.20 * behavior_score
        + 0.10 * availability_score
        + 0.05 * location_score
    )
    return float(np.clip(score, 0.0, 100.0))
