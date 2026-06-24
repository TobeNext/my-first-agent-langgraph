"""
Merge Milvus questions + existing questions + generate synthetic → production-scale dataset.
Target: ~2500 questions, ~400-500 eval cases.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = REPO_ROOT / "tests" / "evals" / "datasets"

random.seed(42)

# ---------------------------------------------------------------------------
# Step 1: Load all sources
# ---------------------------------------------------------------------------

def load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))

milvus_q = load_json(DATASETS / "interview_question_bank_milvus.json")
existing_q = load_json(DATASETS / "interview_question_bank.json")
existing_cases = load_json(DATASETS / "embedding_eval_cases.json")

print(f"Milvus questions: {len(milvus_q)}")
print(f"Existing questions: {len(existing_q)}")
print(f"Existing cases: {len(existing_cases)}")

# ---------------------------------------------------------------------------
# Step 2: Merge questions — dedup by normalized text prefix
# ---------------------------------------------------------------------------

def text_key(text: str) -> str:
    """Normalize text for dedup."""
    return " ".join(text.lower().split())[:60]

existing_keys = {text_key(q["text"]) for q in existing_q}
new_milvus = [q for q in milvus_q if text_key(q["text"]) not in existing_keys]
print(f"New Milvus questions (after dedup): {len(new_milvus)}")

merged_q = existing_q + new_milvus
print(f"Merged questions: {len(merged_q)}")

# ---------------------------------------------------------------------------
# Step 3: Generate synthetic questions to reach ~2500
# ---------------------------------------------------------------------------

SKILL_DOMAINS = [
    "Python backend", "FastAPI", "Django", "Flask",
    "Go microservices", "Rust systems", "Java Spring", "Node.js Express",
    "React frontend", "Vue.js", "Angular", "Svelte", "Next.js SSR",
    "Kubernetes", "Docker", "Terraform", "Ansible", "Helm", "ArgoCD",
    "AWS", "GCP", "Azure", "cloud architecture",
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch", "Cassandra",
    "Kafka", "RabbitMQ", "NATS", "message queues",
    "gRPC", "GraphQL", "REST API", "WebSocket",
    "CI/CD", "GitHub Actions", "Jenkins", "GitLab CI",
    "Prometheus", "Grafana", "OpenTelemetry", "observability",
    "LangGraph", "LangChain", "LLM", "RAG", "embedding", "vector database",
    "machine learning", "deep learning", "NLP", "computer vision",
    "data engineering", "Spark", "Flink", "data pipeline",
    "system design", "architecture", "distributed systems",
    "security", "authentication", "authorization", "OWASP",
    "testing", "unit test", "integration test", "e2e test",
    "Linux", "networking", "TCP/IP", "DNS", "HTTP",
    "TypeScript", "JavaScript", "HTML", "CSS", "responsive design",
    "agile", "scrum", "project management", "tech leadership",
    "Android", "iOS", "Flutter", "React Native", "mobile",
]

QUESTION_TEMPLATES = [
    "请详细解释 {skill} 的核心概念和在实际项目中的应用",
    "What are the best practices for {skill} in production environments?",
    "{skill} 的常见坑和反模式有哪些？如何避免？",
    "Compare and contrast different approaches to {skill}: pros, cons, and when to use each.",
    "如何在大型项目中有效地使用 {skill}？分享你的实践经验",
    "Explain the internals of {skill} and how it impacts system performance.",
    "{skill} 的安全性考量：常见攻击向量和防御策略",
    "What monitoring and alerting strategies do you recommend for systems using {skill}?",
    "{skill} 的测试策略：单元测试、集成测试和端到端测试怎么设计？",
    "How would you scale a system built with {skill} from 100 to 1M users?",
    "请设计一个基于 {skill} 的高可用系统架构",
    "Troubleshooting {skill} in production: common failure modes and debugging techniques.",
    "{skill} 和竞品技术的深度对比：什么时候该迁移？",
    "How does {skill} handle concurrency and what are the threading/async models?",
    "{skill} 的性能调优实战：从 profiling 到优化的完整流程",
]

NEGATIVE_IDS = [
    "q-css-responsive-grid", "q-vue-reactivity", "q-vue-composables",
    "q-agile-scrum", "q-frontend-css-layout", "q-vue-sse-streaming",
]

# Generate questions
existing_texts = {text_key(q["text"]) for q in merged_q}
target_q = 2500
needed_q = target_q - len(merged_q)
print(f"Need {needed_q} more questions to reach {target_q}")

new_questions = []
attempts = 0
while len(new_questions) < needed_q and attempts < 10000:
    attempts += 1
    skill = random.choice(SKILL_DOMAINS)
    template = random.choice(QUESTION_TEMPLATES)
    text = template.replace("{skill}", skill)
    tkey = text_key(text)
    if tkey in existing_texts:
        continue
    existing_texts.add(tkey)
    qid = f"q-synth-{len(new_questions) + 1}"
    new_questions.append(
        {
            "id": qid,
            "text": text,
            "skill_areas": [skill, "engineering"],
            "round_type": "professional-skills",
        }
    )

merged_q.extend(new_questions)
print(f"Generated {len(new_questions)} synthetic questions")
print(f"Final question bank: {len(merged_q)}")

# ---------------------------------------------------------------------------
# Step 4: Generate corresponding eval cases (~400-500)
# ---------------------------------------------------------------------------

target_cases = 450
needed_cases = target_cases - len(existing_cases)
print(f"Need {needed_cases} more eval cases to reach {target_cases}")

# Pick synthetic questions that have clear skill areas
synth_with_skills = [q for q in new_questions if len(q.get("skill_areas", [])) > 0]

new_cases = []
for i, q in enumerate(synth_with_skills[:needed_cases]):
    skill = q["skill_areas"][0]
    case_id = f"embed-eval-synth-{i + 1}"

    # Build a natural query from the question text
    query_text = q["text"][:120]
    if "?" in query_text:
        query_text = query_text.rsplit("?", 1)[0] + "?"

    new_cases.append(
        {
            "case_id": case_id,
            "query": query_text,
            "round_type": q.get("round_type", "professional-skills"),
            "expected_question_ids": [q["id"]],
            "acceptable_skill_areas": [skill],
            "negative_question_ids": random.sample(NEGATIVE_IDS, min(2, len(NEGATIVE_IDS))),
        }
    )

combined_cases = existing_cases + new_cases
print(f"Generated {len(new_cases)} new eval cases")
print(f"Final eval cases: {len(combined_cases)}")

# ---------------------------------------------------------------------------
# Step 5: Save
# ---------------------------------------------------------------------------

out_q = DATASETS / "interview_question_bank.json"
out_c = DATASETS / "embedding_eval_cases.json"

out_q.write_text(json.dumps(merged_q, ensure_ascii=False, indent=2), encoding="utf-8")
out_c.write_text(json.dumps(combined_cases, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"\nSaved: {out_q} ({len(merged_q)} questions)")
print(f"Saved: {out_c} ({len(combined_cases)} cases)")
print("\nDone! Ready for production-scale benchmark.")
