"""
Expand question bank to 10,000 by fetching from GitHub + generating diverse synthetic questions.
Strategy:
  1. Fetch from public GitHub interview-question repos (if accessible)
  2. Fill the rest with template-based generation using 200+ unique domain×template combos
"""

from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = REPO_ROOT / "tests" / "evals" / "datasets"

random.seed(42)

# ---------------------------------------------------------------------------
# Step 1: Load existing
# ---------------------------------------------------------------------------
bank = json.loads((DATASETS / "interview_question_bank.json").read_text(encoding="utf-8"))
cases = json.loads((DATASETS / "embedding_eval_cases.json").read_text(encoding="utf-8"))
print(f"Existing: {len(bank)} questions, {len(cases)} cases")

existing_texts = {" ".join(q["text"].lower().split())[:80] for q in bank}

# ---------------------------------------------------------------------------
# Step 2: Fetch from GitHub public interview question repos
# ---------------------------------------------------------------------------

GITHUB_SOURCES = [
    "https://raw.githubusercontent.com/DopplerHQ/awesome-interview-questions/master/README.md",
    "https://raw.githubusercontent.com/yangshun/tech-interview-handbook/master/contents/algorithms/README.md",
    "https://raw.githubusercontent.com/jwasham/coding-interview-university/master/README.md",
]

def fetch_github_questions() -> list[dict]:
    """Try to fetch real questions from GitHub raw content."""
    import urllib.request
    import urllib.error

    questions = []
    for url in GITHUB_SOURCES:
        try:
            print(f"  Fetching: {url[:60]}...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                content = resp.read().decode("utf-8", errors="ignore")
            # Extract bullet-point questions
            lines = [l.strip() for l in content.splitlines() if l.strip().startswith(("- ", "* ", "1. ", "2. ", "3. "))]
            for line in lines:
                clean = re.sub(r"^[\-\*\d\.]+\s*", "", line).strip()
                if len(clean) > 20 and len(clean) < 300 and "?" in clean:
                    questions.append(clean)
            print(f"    Extracted {len(questions)} question-like lines so far")
        except Exception as exc:
            print(f"    Failed: {exc}")
    return questions


print("\nFetching from GitHub...")
web_raw = fetch_github_questions()
print(f"Got {len(web_raw)} raw question lines from GitHub")

# Convert to bank format
web_questions = []
for i, text in enumerate(web_raw):
    tkey = " ".join(text.lower().split())[:80]
    if tkey in existing_texts:
        continue
    existing_texts.add(tkey)
    web_questions.append({
        "id": f"q-web-{i + 1}",
        "text": text,
        "skill_areas": ["general", "interview"],
        "round_type": "professional-skills",
    })

print(f"New web questions (after dedup): {len(web_questions)}")
bank.extend(web_questions)

# ---------------------------------------------------------------------------
# Step 3: Generate diverse synthetic questions to reach 10,000
# ---------------------------------------------------------------------------

# Rich domain × topic matrix for maximum diversity
DOMAINS = [
    # Backend
    ("Python", ["asyncio", "decorator", "generator", "GIL", "type hints", "context manager", "metaclass", "descriptor", "coroutine", "threading"]),
    ("FastAPI", ["dependency injection", "middleware", "background tasks", "WebSocket", "pagination", "rate limiting", "CORS", "OAuth2", "testing", "deployment"]),
    ("Go", ["goroutine", "channel", "interface", "error handling", "context", "GC", "escape analysis", "embedding", "generics", "testing"]),
    ("Rust", ["ownership", "lifetime", "async", "tokio", "error handling", "traits", "macros", "unsafe", "cargo", "testing"]),
    ("Java", ["Spring Boot", "JVM tuning", "GC algorithms", "concurrency", "Stream API", "JPA", "microservices", "reactive", "testing", "deployment"]),
    ("Node.js", ["event loop", "streams", "cluster", "Express", "NestJS", "middleware", "error handling", "performance", "security", "testing"]),
    # Frontend
    ("React", ["hooks", "context", "memo", "Suspense", "Server Components", "state management", "routing", "testing", "performance", "accessibility"]),
    ("Vue", ["reactivity", "composables", "Pinia", "Vue Router", "slots", "teleport", "transition", "SSR", "testing", "TypeScript"]),
    ("Angular", ["dependency injection", "RxJS", "NgRx", "routing", "forms", "pipes", "directives", "lazy loading", "testing", "SSR"]),
    ("CSS", ["Flexbox", "Grid", "animations", "responsive", "container queries", "layers", "variables", "preprocessors", "Tailwind", "BEM"]),
    # Infrastructure
    ("Kubernetes", ["Pods", "Services", "Deployments", "ConfigMaps", "Secrets", "RBAC", "networking", "storage", "operators", "Helm"]),
    ("Docker", ["multi-stage", "compose", "networking", "volumes", "security", "optimization", "registry", "orchestration", "Swarm", "buildkit"]),
    ("AWS", ["Lambda", "EC2", "S3", "RDS", "DynamoDB", "SQS", "CloudFormation", "IAM", "VPC", "CloudFront"]),
    ("Terraform", ["state", "modules", "workspaces", "providers", "provisioners", "remote backend", "sentinel", "testing", "CI/CD", "drift detection"]),
    # Data
    ("PostgreSQL", ["indexing", "vacuum", "replication", "partitioning", "JSON", "CTE", "window functions", "extensions", "full-text search", "performance"]),
    ("Redis", ["data structures", "persistence", "cluster", "sentinel", "pipelining", "pub/sub", "streams", "Lua scripting", "caching patterns", "eviction"]),
    ("Kafka", ["partitions", "consumer groups", "exactly-once", "compaction", "mirroring", "Connect", "Streams", "monitoring", "tuning", "schema registry"]),
    ("Elasticsearch", ["mapping", "analyzer", "aggregations", "scroll", "reindex", "snapshot", "ILM", "security", "performance", "cluster"]),
    # AI/ML
    ("LangGraph", ["StateGraph", "checkpoint", "nodes", "edges", "streaming", "tools", "human-in-the-loop", "parallel", "memory", "deployment"]),
    ("RAG", ["chunking", "embedding", "retrieval", "reranking", "hybrid search", "evaluation", "prompt", "vector DB", "multi-modal", "streaming"]),
    ("LLM", ["fine-tuning", "prompt engineering", "structured output", "token optimization", "temperature", "safety", "evaluation", "deployment", "cost", "orchestration"]),
    ("Machine Learning", ["overfitting", "feature engineering", "cross-validation", "ensemble", "deployment", "monitoring", "drift", "explainability", "AutoML", "pipeline"]),
    # Cloud Native
    ("Prometheus", ["metrics", "alerting", "recording rules", "federation", "remote write", "service discovery", "TSDB", "QL", "HA", "scaling"]),
    ("OpenTelemetry", ["tracing", "sampling", "propagation", "collector", "SDK", "instrumentation", "exporter", "processor", "semantic conventions", "best practices"]),
    ("CI/CD", ["GitHub Actions", "GitLab CI", "Jenkins", "ArgoCD", "Spinnaker", "canary", "blue-green", "rollback", "secrets", "approval gates"]),
    # System Design
    ("System Design", ["URL shortener", "rate limiter", "chat system", "news feed", "search engine", "CDN", "distributed cache", "message queue", "file storage", "real-time"]),
]

TEMPLATES = [
    "请详细解释 {topic} 在 {domain} 中的核心原理和实际应用",
    "What are the best practices for {topic} in {domain} production environments?",
    "{domain} 中 {topic} 的常见错误和调试方法",
    "Compare {topic} approaches in {domain}: trade-offs and recommendations",
    "如何在大型 {domain} 项目中正确使用 {topic}？",
    "Explain {topic} internals in {domain} and performance implications",
    "{domain} 中 {topic} 的安全最佳实践和常见攻击防御",
    "How to monitor and alert on {topic} in {domain} systems?",
    "{topic} 在 {domain} 中的测试策略：单元、集成和端到端测试",
    "How to scale {topic} in {domain} from prototype to enterprise?",
    "{domain} 中 {topic} 的性能调优：从 profiling 到优化",
    "Troubleshooting {topic} failures in {domain}: systematic approach",
    "{topic} 在 {domain} 微服务架构中的设计模式",
    "How does {domain} implement {topic} differently from competitors?",
    "{domain} {topic} 的成本优化和资源管理策略",
    "Discuss {topic} observability patterns specific to {domain}",
    "{domain} 中 {topic} 的版本兼容和向后兼容策略",
    "Event-driven architecture for {topic} in {domain} systems",
    "{domain} {topic} 的容量规划和自动扩缩容方案",
    "Implementing fault tolerance for {topic} in {domain}",
    "{topic} data consistency patterns in {domain} applications",
    "How to migrate legacy {domain} systems to modern {topic}?",
    "{domain} {topic} 的多租户架构和隔离策略",
    "Real-time {topic} processing in {domain}: stream vs batch",
    "{domain} {topic} 的配置管理和特性开关最佳实践",
    "Serverless architecture for {topic} in {domain}",
    "{domain} {topic} 的 API 设计和版本管理策略",
    "Caching strategies for {topic} in {domain}",
    "{domain} {topic} 的灾难恢复和高可用方案",
    "Multi-region deployment of {topic} in {domain} applications",
]

TARGET = 10000
needed = TARGET - len(bank)
print(f"\nNeed {needed} more questions to reach {TARGET}")

new_questions = []
all_combos = [(d, t) for d, topics in DOMAINS for t in topics for _ in range(3)]
random.shuffle(all_combos)

for domain, topic in all_combos:
    if len(new_questions) >= needed:
        break
    template = random.choice(TEMPLATES)
    text = template.format(domain=domain, topic=topic)
    tkey = " ".join(text.lower().split())[:80]
    if tkey in existing_texts:
        continue
    existing_texts.add(tkey)
    new_questions.append({
        "id": f"q-10k-{len(new_questions) + 1}",
        "text": text,
        "skill_areas": [domain, topic],
        "round_type": "professional-skills",
    })

print(f"Generated {len(new_questions)} new synthetic questions")
bank.extend(new_questions)

# ---------------------------------------------------------------------------
# Step 4: Generate matching eval cases (~1 per 10 questions)
# ---------------------------------------------------------------------------
neg_ids = ["q-css-responsive-grid", "q-vue-reactivity", "q-agile-scrum", "q-frontend-css-layout"]
case_ids = {c["case_id"] for c in cases}

target_cases = 1000
needed_cases = target_cases - len(cases)
synth_qs = new_questions[:needed_cases]

new_cases = []
for q in synth_qs:
    cid = f"embed-eval-10k-{len(new_cases) + 1}"
    if cid in case_ids:
        continue
    skill = q.get("skill_areas", ["general"])[0]
    new_cases.append({
        "case_id": cid,
        "query": q["text"][:150],
        "round_type": q.get("round_type", "professional-skills"),
        "expected_question_ids": [q["id"]],
        "acceptable_skill_areas": [skill],
        "negative_question_ids": random.sample(neg_ids, 2),
    })

cases.extend(new_cases)
print(f"Generated {len(new_cases)} new eval cases")

# Save
(DATASETS / "interview_question_bank.json").write_text(
    json.dumps(bank, ensure_ascii=False, indent=2), encoding="utf-8"
)
(DATASETS / "embedding_eval_cases.json").write_text(
    json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8"
)

print(f"\n✅ Final: {len(bank)} questions, {len(cases)} cases")
print(f"Sources: Milvus(125) + Manual(264) + GitHub({len(web_questions)}) + Synthetic({len(new_questions)})")
