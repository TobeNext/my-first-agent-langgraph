from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedResumeMarkdown:
    professionalSkillsSection: str
    projectExperienceSection: str
    normalizedSkills: list[str]
    normalizedProjectTopics: list[str]
    warnings: list[str]
    validationErrors: list[str]


SECTION_HEADINGS = {
    "professionalSkills": {
        "专业技能",
        "核心技能",
        "技术栈",
        "技能清单",
        "技能栈",
        "技术能力",
        "skills",
        "skill set",
        "core skills",
        "technical skills",
        "professional skills",
    },
    "projectExperience": {
        "项目经历",
        "项目经验",
        "项目实践",
        "项目案例",
        "代表项目",
        "projects",
        "project experience",
        "project experiences",
        "selected projects",
        "project highlights",
    },
}


def parse_resume_sections(sections: dict[str, str]) -> ParsedResumeMarkdown:
    professional = sections.get("professionalSkills", "").strip()
    project = sections.get("projectExperience", "").strip()
    return _build_result(professional, project, [], [])


def parse_resume_markdown(markdown: str) -> ParsedResumeMarkdown:
    if not markdown.strip():
        return _build_result("", "", [], ["简历内容不能为空。"])

    collected: dict[str, list[str]] = {"professionalSkills": [], "projectExperience": []}
    active_key: str | None = None
    warnings: list[str] = []
    errors: list[str] = []

    for line in markdown.splitlines():
        heading = _extract_heading_name(line)
        section_key = _resolve_section_key(heading) if heading else None
        if section_key:
            active_key = section_key
            continue
        if heading and line.lstrip().startswith("#"):
            active_key = None
            continue
        if active_key:
            collected[active_key].append(line)

    professional = "\n".join(collected["professionalSkills"]).strip()
    project = "\n".join(collected["projectExperience"]).strip()
    if not professional:
        errors.append("缺少章节：### 专业技能。")
    if not project:
        errors.append("缺少章节：### 项目经历。")
    return _build_result(professional, project, warnings, errors)


def extract_normalized_resume_topics(section_markdown: str) -> list[str]:
    lines = [
        line.strip()
        for line in section_markdown.splitlines()
        if line.strip() and line.strip() != "..."
    ]
    grouped_lines: list[str] = []
    current_group: list[str] = []
    has_structured_items = False

    for line in lines:
        if _is_list_item_line(line):
            has_structured_items = True
            if current_group:
                grouped_lines.append(" ".join(current_group))
            content = _get_list_item_content(line)
            current_group = [content] if content else []
            continue
        if has_structured_items:
            if current_group:
                current_group.append(re.sub(r"\s+", " ", line).strip())
            continue
        grouped_lines.append(re.sub(r"\s+", " ", line).strip())

    if current_group:
        grouped_lines.append(" ".join(current_group))

    result: list[str] = []
    seen: set[str] = set()
    for line in grouped_lines:
        normalized = re.sub(r"\s+", " ", line).strip()
        key = normalized.lower()
        if len(normalized) > 1 and key not in seen:
            seen.add(key)
            result.append(normalized)
    return result[:8]


def _build_result(
    professional: str,
    project: str,
    warnings: list[str],
    errors: list[str],
) -> ParsedResumeMarkdown:
    return ParsedResumeMarkdown(
        professionalSkillsSection=professional.strip(),
        projectExperienceSection=project.strip(),
        normalizedSkills=extract_normalized_resume_topics(professional),
        normalizedProjectTopics=extract_normalized_resume_topics(project),
        warnings=warnings,
        validationErrors=errors,
    )


def _extract_heading_name(line: str) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None
    markdown_match = re.match(r"^#{1,6}\s*(.+?)\s*#*\s*$", stripped)
    if markdown_match:
        return markdown_match.group(1).strip()
    bold_match = re.match(r"^\*\*(.+?)\*\*$", stripped)
    if bold_match:
        return bold_match.group(1).strip()
    return stripped


def _resolve_section_key(heading: str | None) -> str | None:
    if not heading:
        return None
    normalized = _normalize_heading(heading)
    for key, headings in SECTION_HEADINGS.items():
        if normalized in {_normalize_heading(item) for item in headings}:
            return key
    return None


def _normalize_heading(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().strip(":：- ")).lower()


def _is_list_item_line(value: str) -> bool:
    return bool(re.match(r"^(?:[-*+•]\s+|\d+[.)]\s+)", value.lstrip()))


def _get_list_item_content(value: str) -> str:
    return re.sub(r"^(?:[-*+•]\s+|\d+[.)]\s+)", "", value.lstrip()).strip()
