from typing import Literal

from pydantic import Field, model_validator

from app.schemas.interview_state import ContractModel, InterviewSystemSettings

INTERVIEW_START_REQUEST_KIND = "interview-start"
INTERVIEW_START_PROTOCOL_VERSION = "2026-05-structured-start-v1"


class InterviewResumeSections(ContractModel):
    professionalSkills: str
    projectExperience: str


class InterviewStartRequest(ContractModel):
    requestKind: Literal["interview-start"]
    protocolVersion: Literal["2026-05-structured-start-v1"]
    startInterview: Literal[True]
    threadId: str = Field(min_length=1)
    userId: str | None = Field(default=None, min_length=1)
    resumeMarkdown: str = Field(min_length=1)
    jobDescriptionMarkdown: str = ""
    settings: InterviewSystemSettings
    resumeSections: InterviewResumeSections | None = None

    @model_validator(mode="after")
    def validate_round_settings(self) -> "InterviewStartRequest":
        settings = self.settings
        if settings.skipProfessionalSkillsRound and settings.skipProjectExperienceRound:
            raise ValueError(
                "Professional skills and project experience rounds cannot both be skipped."
            )
        if not settings.skipProfessionalSkillsRound and settings.professionalQuestionCount < 1:
            raise ValueError("Professional skills question count must be at least 1.")
        if settings.skipProfessionalSkillsRound and settings.professionalQuestionCount != 0:
            raise ValueError("Professional skills question count must be 0 when skipped.")
        if not settings.skipProjectExperienceRound and settings.projectQuestionCount < 1:
            raise ValueError("Project experience question count must be at least 1.")
        if settings.skipProjectExperienceRound and settings.projectQuestionCount != 0:
            raise ValueError("Project experience question count must be 0 when skipped.")
        return self
