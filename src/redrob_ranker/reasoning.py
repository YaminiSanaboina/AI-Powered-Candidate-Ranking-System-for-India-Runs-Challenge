from .scoring import SKILL_KEYWORDS


def build_reasoning(candidate):
    years = candidate.get("years_of_experience", 0.0)
    title = candidate.get("current_title", "candidate").strip()
    matched_skills = candidate.get("matched_skills", [])
    recruiter_response_rate = candidate.get("recruiter_response_rate", 0.0)
    interview_completion_rate = candidate.get("interview_completion_rate", 0.0)
    github_activity_score = candidate.get("github_activity_score", 0.0)
    saved_by_recruiters_30d = candidate.get("saved_by_recruiters_30d", 0)
    open_to_work_flag = candidate.get("open_to_work_flag", False)
    notice_period_days = candidate.get("notice_period_days", None)

    experience_part = f"{years:.1f} yrs exp"
    top_skills = sorted(
        matched_skills,
        key=lambda skill: list(SKILL_KEYWORDS).index(skill) if skill in SKILL_KEYWORDS else len(SKILL_KEYWORDS),
    )
    skills_part = ", ".join(top_skills[:3]) if top_skills else "no core AI skills"

    strengths = []
    if recruiter_response_rate >= 0.7:
        strengths.append(f"high recruiter response {recruiter_response_rate:.2f}")
    if interview_completion_rate >= 0.7:
        strengths.append(f"strong interview completion {interview_completion_rate:.2f}")
    if github_activity_score >= 60.0:
        strengths.append(f"active GitHub {github_activity_score:.1f}")
    if saved_by_recruiters_30d >= 3:
        strengths.append(f"saved by recruiters {int(saved_by_recruiters_30d)} times")

    if not strengths:
        strengths.append(
            f"response {recruiter_response_rate:.2f}, interview completion {interview_completion_rate:.2f}"
        )

    availability_part = "open to work" if open_to_work_flag else "not actively open"
    notice_part = f"notice {int(notice_period_days)}d" if notice_period_days is not None else "notice unknown"

    return (
        f"{title} with {experience_part}; strongest skills: {skills_part}; "
        f"{' ; '.join(strengths)}; {availability_part}, {notice_part}."
    )
