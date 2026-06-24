"""
Expand interview question bank and eval cases to 4x size.
Generates new synthetic entries, appends to originals, writes output.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = REPO_ROOT / "tests" / "evals" / "datasets"

# ---------------------------------------------------------------------------
# New question bank entries (~300 new)
# ---------------------------------------------------------------------------

NEW_QUESTIONS = [
    # --- Go ---
    {"id": "q-go-goroutine", "text": "Go 语言的 goroutine 和 channel 的协作模型是怎样的？如何避免 goroutine 泄漏？", "skill_areas": ["Go", "concurrency", "backend"], "round_type": "professional-skills"},
    {"id": "q-go-interface", "text": "Go interface 的隐式实现和空接口 interface{} 在设计模式中的应用场景", "skill_areas": ["Go", "interface", "design patterns"], "round_type": "professional-skills"},
    {"id": "q-go-memory", "text": "Go 的逃逸分析（escape analysis）和垃圾回收（GC）机制如何影响性能？", "skill_areas": ["Go", "memory", "GC"], "round_type": "professional-skills"},
    {"id": "q-go-context", "text": "Explain Go context package: cancellation, deadlines, and how to propagate context through a microservice call chain.", "skill_areas": ["Go", "context", "microservices"], "round_type": "professional-skills"},
    {"id": "q-go-vs-python", "text": "Go 和 Python 在后端服务中的选型对比：性能、生态、开发效率各有什么优劣？", "skill_areas": ["Go", "Python", "backend"], "round_type": "professional-skills"},

    # --- Rust ---
    {"id": "q-rust-ownership", "text": "Rust 的所有权（ownership）、借用（borrowing）和生命周期（lifetime）的核心规则", "skill_areas": ["Rust", "memory safety", "systems"], "round_type": "professional-skills"},
    {"id": "q-rust-async", "text": "Rust 的 async/await 与 tokio runtime 的工作原理，和 Go goroutine 有什么区别？", "skill_areas": ["Rust", "async", "concurrency"], "round_type": "professional-skills"},
    {"id": "q-rust-error", "text": "Rust 的错误处理：Result<T,E>、? 操作符、anyhow vs thiserror 的使用场景", "skill_areas": ["Rust", "error handling", "systems"], "round_type": "professional-skills"},

    # --- Kubernetes ---
    {"id": "q-k8s-architecture", "text": "Kubernetes 核心架构：API Server、Scheduler、Controller Manager、etcd 的协作流程", "skill_areas": ["Kubernetes", "orchestration", "cloud native"], "round_type": "professional-skills"},
    {"id": "q-k8s-pod-lifecycle", "text": "Kubernetes Pod 生命周期和探针（liveness/readiness/startup probe）的最佳实践", "skill_areas": ["Kubernetes", "Pod", "DevOps"], "round_type": "professional-skills"},
    {"id": "q-k8s-networking", "text": "Kubernetes 网络模型：CNI、Service、Ingress、NetworkPolicy 如何协同工作？", "skill_areas": ["Kubernetes", "networking", "cloud native"], "round_type": "professional-skills"},
    {"id": "q-k8s-helm", "text": "Helm Chart 的最佳实践：如何管理多环境配置、依赖和版本升级？", "skill_areas": ["Kubernetes", "Helm", "DevOps"], "round_type": "professional-skills"},
    {"id": "q-k8s-operator", "text": "What is a Kubernetes Operator and when should you build one instead of using Helm?", "skill_areas": ["Kubernetes", "Operator", "cloud native"], "round_type": "professional-skills"},

    # --- React ---
    {"id": "q-react-hooks", "text": "React Hooks 的原理：useState、useEffect、useMemo、useCallback 的依赖数组和闭包陷阱", "skill_areas": ["React", "Hooks", "frontend"], "round_type": "professional-skills"},
    {"id": "q-react-state", "text": "React 状态管理方案对比：Redux Toolkit、Zustand、Jotai、React Query 各自适用什么场景？", "skill_areas": ["React", "state management", "frontend"], "round_type": "professional-skills"},
    {"id": "q-react-rendering", "text": "React 渲染优化：React.memo、useMemo、虚拟列表、代码分割的实际应用", "skill_areas": ["React", "performance", "frontend"], "round_type": "professional-skills"},
    {"id": "q-react-ssr", "text": "React SSR（Next.js）和 CSR 的取舍：SEO、首屏性能、开发体验如何权衡？", "skill_areas": ["React", "SSR", "Next.js"], "round_type": "professional-skills"},
    {"id": "q-react-testing", "text": "React Testing Library vs Enzyme: how to write maintainable component tests?", "skill_areas": ["React", "testing", "frontend"], "round_type": "professional-skills"},

    # --- ML / AI ---
    {"id": "q-ml-overfitting", "text": "机器学习中的过拟合（overfitting）如何诊断和解决？正则化、dropout、early stopping 的原理", "skill_areas": ["machine learning", "overfitting", "AI"], "round_type": "professional-skills"},
    {"id": "q-ml-transformer", "text": "Transformer 架构的 Self-Attention 机制详解：Q、K、V 的计算过程和 Multi-Head Attention", "skill_areas": ["machine learning", "Transformer", "NLP"], "round_type": "professional-skills"},
    {"id": "q-ml-finetune", "text": "大模型微调（fine-tuning）方法：LoRA、QLoRA、P-Tuning 的原理和适用场景", "skill_areas": ["machine learning", "fine-tuning", "LLM"], "round_type": "professional-skills"},
    {"id": "q-ml-embedding", "text": "文本 embedding 模型的训练目标：对比学习（contrastive learning）和 InfoNCE loss 的数学原理", "skill_areas": ["machine learning", "embedding", "NLP"], "round_type": "professional-skills"},
    {"id": "q-ml-eval", "text": "How do you evaluate an LLM's output quality? BLEU, ROUGE, BERTScore, and human evaluation trade-offs.", "skill_areas": ["machine learning", "evaluation", "LLM"], "round_type": "professional-skills"},
    {"id": "q-ml-rag-vs-finetune", "text": "RAG vs Fine-tuning: 什么时候该用 RAG 增强，什么时候该微调模型？各自的成本和效果如何？", "skill_areas": ["machine learning", "RAG", "fine-tuning"], "round_type": "professional-skills"},
    {"id": "q-ml-vector-db", "text": "向量数据库选型：Milvus、Qdrant、Pinecone、Weaviate 的功能对比和适用场景", "skill_areas": ["machine learning", "vector database", "RAG"], "round_type": "professional-skills"},

    # --- Data Engineering ---
    {"id": "q-de-spark", "text": "Apache Spark 的 RDD、DataFrame、Dataset 有什么区别？什么场景用哪个？", "skill_areas": ["data engineering", "Spark", "big data"], "round_type": "professional-skills"},
    {"id": "q-de-kafka", "text": "Kafka 的 partition、consumer group、offset 机制如何保证消息的可靠性和顺序性？", "skill_areas": ["data engineering", "Kafka", "streaming"], "round_type": "professional-skills"},
    {"id": "q-de-etl", "text": "ETL vs ELT 架构：在现代数据栈中 Airbyte、dbt、Snowflake 如何协作？", "skill_areas": ["data engineering", "ETL", "data pipeline"], "round_type": "professional-skills"},
    {"id": "q-de-flink", "text": "Flink 的 exactly-once 语义和 checkpoint 机制如何实现？和 Spark Streaming 有什么不同？", "skill_areas": ["data engineering", "Flink", "streaming"], "round_type": "professional-skills"},
    {"id": "q-de-lakehouse", "text": "Data Lakehouse 架构：Delta Lake、Iceberg、Hudi 的技术对比和选型建议", "skill_areas": ["data engineering", "lakehouse", "architecture"], "round_type": "professional-skills"},

    # --- More depth in existing areas ---
    {"id": "q-vue-teleport", "text": "Vue 3 Teleport 和 Suspense 组件的原理和使用场景", "skill_areas": ["Vue", "components", "frontend"], "round_type": "professional-skills"},
    {"id": "q-vue-custom-ref", "text": "Vue 3 自定义 ref（customRef）和 shallowRef 的底层实现原理", "skill_areas": ["Vue", "reactivity", "frontend"], "round_type": "professional-skills"},
    {"id": "q-nestjs-microservice", "text": "NestJS 微服务模式：TCP、Redis、gRPC、Kafka transport 的区别和配置", "skill_areas": ["NestJS", "microservices", "backend"], "round_type": "professional-skills"},
    {"id": "q-nestjs-cqrs", "text": "NestJS CQRS 模式：Command、Query、Event Bus 的实现和 Saga 编排", "skill_areas": ["NestJS", "CQRS", "architecture"], "round_type": "professional-skills"},
    {"id": "q-python-metaclass", "text": "Python 元类（metaclass）的实现原理：__new__、__init__、__call__ 的执行顺序", "skill_areas": ["Python", "metaclass", "advanced"], "round_type": "professional-skills"},
    {"id": "q-python-c-ext", "text": "Python C 扩展和 Cython 的性能优化：什么时候该用原生扩展替代纯 Python？", "skill_areas": ["Python", "performance", "C extension"], "round_type": "professional-skills"},
    {"id": "q-fastapi-websocket", "text": "FastAPI WebSocket 端点实现：连接管理、心跳检测和广播机制", "skill_areas": ["FastAPI", "WebSocket", "real-time"], "round_type": "professional-skills"},
    {"id": "q-fastapi-testing", "text": "FastAPI TestClient 和 pytest 集成：如何编写依赖注入 mock 和异步测试？", "skill_areas": ["FastAPI", "testing", "pytest"], "round_type": "professional-skills"},
    {"id": "q-langgraph-human-loop", "text": "LangGraph 的 human-in-the-loop 模式：interrupt、Command、动态 breakpoint 的实现", "skill_areas": ["LangGraph", "human-in-the-loop", "agent"], "round_type": "professional-skills"},
    {"id": "q-langgraph-parallel", "text": "LangGraph 的并行节点（Send API）和 map-reduce 模式如何实现？", "skill_areas": ["LangGraph", "parallel", "graph"], "round_type": "professional-skills"},

    # --- Network / Protocol ---
    {"id": "q-tcp-ip", "text": "TCP 三次握手和四次挥手的过程详解：为什么需要 TIME_WAIT 状态？", "skill_areas": ["network", "TCP", "protocol"], "round_type": "professional-skills"},
    {"id": "q-dns-cdn", "text": "DNS 解析和 CDN 加速原理：Anycast、EDNS Client Subnet 如何优化全球用户访问？", "skill_areas": ["network", "DNS", "CDN"], "round_type": "professional-skills"},
    {"id": "q-tls-handshake", "text": "TLS 1.3 握手过程：相比 TLS 1.2 减少了哪些步骤？0-RTT 的风险是什么？", "skill_areas": ["network", "TLS", "security"], "round_type": "professional-skills"},

    # --- OS / Linux ---
    {"id": "q-linux-process", "text": "Linux 进程管理和调度：CFS 调度器、nice 值、cgroup 的资源限制机制", "skill_areas": ["Linux", "process", "OS"], "round_type": "professional-skills"},
    {"id": "q-linux-memory", "text": "Linux 虚拟内存管理：页表、TLB、swap、OOM killer 的工作原理", "skill_areas": ["Linux", "memory", "OS"], "round_type": "professional-skills"},
    {"id": "q-linux-io", "text": "Linux I/O 模型：select、poll、epoll、io_uring 的演进和性能对比", "skill_areas": ["Linux", "I/O", "performance"], "round_type": "professional-skills"},

    # --- Algorithms ---
    {"id": "q-algo-hash", "text": "哈希表的实现原理：开放寻址法 vs 链地址法，如何设计一个工业级哈希表？", "skill_areas": ["algorithms", "hash table", "data structures"], "round_type": "professional-skills"},
    {"id": "q-algo-sort", "text": "排序算法对比：快排、归并、堆排的时间复杂度、空间复杂度和稳定性分析", "skill_areas": ["algorithms", "sorting", "data structures"], "round_type": "professional-skills"},
    {"id": "q-algo-tree", "text": "B+Tree 和 LSM-Tree 在数据库索引中的应用：为什么 MySQL 用 B+Tree 而 RocksDB 用 LSM？", "skill_areas": ["algorithms", "tree", "database"], "round_type": "professional-skills"},
    {"id": "q-algo-consensus", "text": "分布式一致性算法：Paxos vs Raft 的原理和实际工程应用", "skill_areas": ["algorithms", "distributed", "consensus"], "round_type": "professional-skills"},

    # --- More project experience ---
    {"id": "project-go-microservice", "text": "请分享你用 Go 构建微服务的经验：如何设计 API、处理错误和做可观测性？", "skill_areas": ["Go", "microservices", "project experience"], "round_type": "project-experience"},
    {"id": "project-k8s-migration", "text": "Describe a Kubernetes migration project you led or participated in: challenges with stateful services, networking, and monitoring.", "skill_areas": ["Kubernetes", "migration", "project experience"], "round_type": "project-experience"},
    {"id": "project-ml-pipeline", "text": "分享你搭建 ML pipeline 的经验：数据收集、特征工程、模型训练、在线推理的完整链路", "skill_areas": ["machine learning", "pipeline", "project experience"], "round_type": "project-experience"},
    {"id": "project-react-architecture", "text": "Describe how you architected a large React application: component design, state management, and performance optimization.", "skill_areas": ["React", "architecture", "project experience"], "round_type": "project-experience"},
    {"id": "project-rust-performance", "text": "请介绍你使用 Rust 优化性能瓶颈的真实案例：profiling 方法、优化策略和最终效果", "skill_areas": ["Rust", "performance", "project experience"], "round_type": "project-experience"},

    # --- Software Engineering ---
    {"id": "q-tdd-practice", "text": "TDD（测试驱动开发）在实际项目中的实践：先写测试再写代码真的可行吗？", "skill_areas": ["TDD", "testing", "engineering"], "round_type": "professional-skills"},
    {"id": "q-refactoring", "text": "代码重构的策略：什么时候该重构？如何在不影响业务的前提下安全重构？", "skill_areas": ["refactoring", "code quality", "engineering"], "round_type": "professional-skills"},
    {"id": "q-tech-debt", "text": "技术债务管理：如何量化、优先级排序和推动偿还技术债务？", "skill_areas": ["tech debt", "engineering management", "quality"], "round_type": "professional-skills"},
    {"id": "q-oncall", "text": "On-call 值班和事故响应：如何建立有效的 incident response 流程和 postmortem 文化？", "skill_areas": ["SRE", "incident response", "DevOps"], "round_type": "professional-skills"},

    # --- More database ---
    {"id": "q-postgres-vs-mysql", "text": "PostgreSQL vs MySQL 的深度对比：MVCC 实现、索引类型、扩展性的差异", "skill_areas": ["database", "PostgreSQL", "MySQL"], "round_type": "professional-skills"},
    {"id": "q-mongodb-aggregation", "text": "MongoDB aggregation pipeline 的最佳实践：如何优化复杂聚合查询的性能？", "skill_areas": ["MongoDB", "NoSQL", "database"], "round_type": "professional-skills"},
    {"id": "q-redis-cluster", "text": "Redis Cluster 的数据分片和故障转移机制：hash slot、gossip 协议和主从切换", "skill_areas": ["Redis", "cluster", "distributed"], "round_type": "professional-skills"},
    {"id": "q-database-sharding", "text": "数据库分库分表策略：垂直拆分 vs 水平拆分、跨分片查询和分布式事务的解决方案", "skill_areas": ["database", "sharding", "distributed"], "round_type": "professional-skills"},

    # --- More observability ---
    {"id": "q-metrics-design", "text": "可观测性指标设计：RED（Rate/Error/Duration）和 USE（Utilization/Saturation/Error）方法论", "skill_areas": ["observability", "metrics", "SRE"], "round_type": "professional-skills"},
    {"id": "q-distributed-tracing", "text": "分布式追踪的实现：Trace Context Propagation（W3C TraceContext）和采样策略", "skill_areas": ["observability", "tracing", "distributed"], "round_type": "professional-skills"},
    {"id": "q-alerting-strategy", "text": "告警策略设计：如何避免告警疲劳？什么指标需要设置告警，什么阈值合理？", "skill_areas": ["observability", "alerting", "SRE"], "round_type": "professional-skills"},

    # --- Additional security & compliance ---
    {"id": "q-owasp-top10", "text": "OWASP Top 10 安全风险在实际 Web 应用中的防护措施", "skill_areas": ["security", "OWASP", "web"], "round_type": "professional-skills"},
    {"id": "q-cors-csrf", "text": "CORS 和 CSRF 的原理及防御：SameSite Cookie、CSRF Token 和 CORS 预检请求", "skill_areas": ["security", "CORS", "CSRF"], "round_type": "professional-skills"},
    {"id": "q-secrets-management", "text": "密钥管理最佳实践：Hashicorp Vault、AWS Secrets Manager、环境变量注入的安全考量", "skill_areas": ["security", "secrets", "DevOps"], "round_type": "professional-skills"},

    # --- More TypeScript ---
    {"id": "q-ts-advanced-types", "text": "TypeScript 高级类型体操：infer、extends 条件类型和模板字面量类型的高级应用", "skill_areas": ["TypeScript", "types", "frontend"], "round_type": "professional-skills"},
    {"id": "q-ts-monorepo", "text": "TypeScript monorepo 管理：pnpm workspace、tsconfig references 和构建优化", "skill_areas": ["TypeScript", "monorepo", "tooling"], "round_type": "professional-skills"},
    {"id": "q-ts-decorator", "text": "TypeScript 装饰器的实现原理：类装饰器、方法装饰器和 reflect-metadata 的协作", "skill_areas": ["TypeScript", "decorator", "metaprogramming"], "round_type": "professional-skills"},

    # --- CI/CD & DevOps ---
    {"id": "q-github-actions", "text": "GitHub Actions 的高级用法：matrix build、reusable workflow、OIDC 认证", "skill_areas": ["CI/CD", "GitHub Actions", "DevOps"], "round_type": "professional-skills"},
    {"id": "q-terraform", "text": "Terraform IaC 最佳实践：state 管理、module 设计、workspace 和环境隔离", "skill_areas": ["Terraform", "IaC", "DevOps"], "round_type": "professional-skills"},
    {"id": "q-blue-green", "text": "蓝绿部署和金丝雀发布：Kubernetes 中如何实现零停机部署和流量切换", "skill_areas": ["DevOps", "deployment", "Kubernetes"], "round_type": "professional-skills"},
    {"id": "q-chaos-engineering", "text": "混沌工程（Chaos Engineering）的实践：如何在生产环境中安全地进行故障注入？", "skill_areas": ["Chaos Engineering", "resilience", "SRE"], "round_type": "professional-skills"},

    # --- Architecture ---
    {"id": "q-cqrs-event-sourcing", "text": "CQRS 和 Event Sourcing 架构的实现细节：事件存储、投影（projection）和最终一致性", "skill_areas": ["CQRS", "Event Sourcing", "architecture"], "round_type": "professional-skills"},
    {"id": "q-hexagonal-arch", "text": "六边形架构（端口和适配器模式）在 Python 项目中的实践：如何解耦业务逻辑和基础设施？", "skill_areas": ["hexagonal architecture", "clean architecture", "design"], "round_type": "professional-skills"},
    {"id": "q-api-gateway", "text": "API Gateway 模式：Kong、APISIX、Envoy 的功能对比和自研 vs 开源的选择", "skill_areas": ["API Gateway", "architecture", "microservices"], "round_type": "professional-skills"},
    {"id": "q-service-mesh", "text": "Service Mesh（Istio/Linkerd）的核心功能：流量管理、安全、可观测性的实现原理", "skill_areas": ["Service Mesh", "Istio", "cloud native"], "round_type": "professional-skills"},

    # --- Team & Management ---
    {"id": "q-tech-lead", "text": "作为 Tech Lead，如何做技术决策？如何平衡技术理想主义和业务交付压力？", "skill_areas": ["tech leadership", "decision making", "management"], "round_type": "professional-skills"},
    {"id": "q-mentoring", "text": "How do you mentor junior engineers effectively? Share your approach to code review and knowledge sharing.", "skill_areas": ["mentoring", "engineering culture", "leadership"], "round_type": "professional-skills"},
    {"id": "q-cross-team", "text": "跨团队协作的挑战：如何推动跨部门的技术项目？如何处理资源冲突和优先级分歧？", "skill_areas": ["collaboration", "project management", "leadership"], "round_type": "professional-skills"},

    # ===== BATCH 2: Extra depth =====

    # --- Frontend depth ---
    {"id": "q-vue-transition", "text": "Vue Transition 和 TransitionGroup 的动画原理和 JavaScript 钩子的使用", "skill_areas": ["Vue", "animation", "frontend"], "round_type": "professional-skills"},
    {"id": "q-vue-slots", "text": "Vue 插槽（slots）的高级用法：作用域插槽、动态插槽名和渲染函数中的插槽", "skill_areas": ["Vue", "slots", "components"], "round_type": "professional-skills"},
    {"id": "q-react-fiber", "text": "React Fiber 架构的原理：可中断渲染、优先级调度和并发模式", "skill_areas": ["React", "Fiber", "architecture"], "round_type": "professional-skills"},
    {"id": "q-react-server-components", "text": "React Server Components vs Client Components: when to use which in Next.js App Router?", "skill_areas": ["React", "RSC", "Next.js"], "round_type": "professional-skills"},
    {"id": "q-frontend-bundler", "text": "前端构建工具对比：Webpack、Vite、Turbopack、esbuild 的实现原理和性能差异", "skill_areas": ["frontend", "bundler", "tooling"], "round_type": "professional-skills"},
    {"id": "q-css-container-query", "text": "CSS Container Queries 和 @layer 的新特性：如何替代传统媒体查询实现组件级响应式？", "skill_areas": ["CSS", "responsive", "frontend"], "round_type": "professional-skills"},
    {"id": "q-wasm-frontend", "text": "WebAssembly 在前端的应用场景：图像处理、加密计算和性能关键路径的优化", "skill_areas": ["WebAssembly", "frontend", "performance"], "round_type": "professional-skills"},

    # --- Backend depth ---
    {"id": "q-graphql-nplusone", "text": "GraphQL 的 N+1 查询问题：DataLoader 的批处理和缓存机制如何解决？", "skill_areas": ["GraphQL", "performance", "backend"], "round_type": "professional-skills"},
    {"id": "q-grpc-vs-rest", "text": "gRPC vs REST 的深度对比：protobuf 序列化、HTTP/2 多路复用和流式传输", "skill_areas": ["gRPC", "REST", "API design"], "round_type": "professional-skills"},
    {"id": "q-idempotent-api", "text": "幂等性 API 的设计：乐观锁、幂等键和去重表在前后端协作中的实践", "skill_areas": ["API design", "idempotency", "backend"], "round_type": "professional-skills"},
    {"id": "q-backpressure", "text": "背压（Backpressure）在流式系统中的实现：Reactive Streams 规范和实际应用", "skill_areas": ["backpressure", "streaming", "reactive"], "round_type": "professional-skills"},
    {"id": "q-circuit-breaker", "text": "熔断器（Circuit Breaker）模式的三种状态和半开探测的实现策略", "skill_areas": ["circuit breaker", "resilience", "microservices"], "round_type": "professional-skills"},

    # --- AI/ML depth ---
    {"id": "q-ml-rlhf", "text": "RLHF（基于人类反馈的强化学习）的原理：reward model 训练和 PPO 优化过程", "skill_areas": ["machine learning", "RLHF", "LLM"], "round_type": "professional-skills"},
    {"id": "q-ml-quantization", "text": "模型量化技术：GPTQ、AWQ、GGUF 的原理和推理性能对比", "skill_areas": ["machine learning", "quantization", "LLM"], "round_type": "professional-skills"},
    {"id": "q-ml-moe", "text": "MoE（混合专家）架构的原理：路由机制、负载均衡和训练稳定性", "skill_areas": ["machine learning", "MoE", "architecture"], "round_type": "professional-skills"},
    {"id": "q-ml-prompt-optimization", "text": "Prompt 自动优化方法：DSPy、APE、OPRO 的原理和实际效果对比", "skill_areas": ["machine learning", "prompt engineering", "LLM"], "round_type": "professional-skills"},
    {"id": "q-ml-agent-framework", "text": "AI Agent 框架对比：LangChain、LangGraph、CrewAI、AutoGen 的架构差异和适用场景", "skill_areas": ["AI agent", "framework", "LLM"], "round_type": "professional-skills"},
    {"id": "q-ml-multimodal", "text": "多模态模型的架构：CLIP、LLaVA 等如何实现文本和图像的联合理解？", "skill_areas": ["multimodal", "CLIP", "deep learning"], "round_type": "professional-skills"},

    # --- Infrastructure depth ---
    {"id": "q-istio-deep", "text": "Istio 的 Sidecar 模式 vs Ambient Mesh：无 Sidecar 的服务网格有什么优势？", "skill_areas": ["Istio", "Service Mesh", "cloud native"], "round_type": "professional-skills"},
    {"id": "q-argo-cd", "text": "Argo CD 的 GitOps 工作流：Application、ApplicationSet 和自动同步策略", "skill_areas": ["Argo CD", "GitOps", "DevOps"], "round_type": "professional-skills"},
    {"id": "q-prometheus-deep", "text": "Prometheus 的存储引擎 TSDB：WAL、compaction 和 retention 机制", "skill_areas": ["Prometheus", "TSDB", "observability"], "round_type": "professional-skills"},
    {"id": "q-loki", "text": "Grafana Loki 的日志聚合架构：ingester、distributor 和标签索引的设计原理", "skill_areas": ["Loki", "logging", "observability"], "round_type": "professional-skills"},
    {"id": "q-envoy-proxy", "text": "Envoy Proxy 的 xDS 动态配置协议：LDS、RDS、CDS、EDS 的协作流程", "skill_areas": ["Envoy", "proxy", "cloud native"], "round_type": "professional-skills"},

    # --- Testing depth ---
    {"id": "q-e2e-playwright", "text": "Playwright vs Cypress vs Selenium: 端到端测试框架的架构对比和选择", "skill_areas": ["testing", "e2e", "Playwright"], "round_type": "professional-skills"},
    {"id": "q-contract-test", "text": "契约测试（Contract Testing）和 Pact 框架：如何在微服务中保证接口兼容性？", "skill_areas": ["testing", "contract test", "microservices"], "round_type": "professional-skills"},
    {"id": "q-snapshot-test", "text": "快照测试（Snapshot Testing）在前端和 API 中的应用：优点、陷阱和最佳实践", "skill_areas": ["testing", "snapshot", "quality"], "round_type": "professional-skills"},
    {"id": "q-fuzzing", "text": "Fuzz Testing 原理：libFuzzer 和 AFL 的覆盖率引导变异策略", "skill_areas": ["testing", "fuzzing", "security"], "round_type": "professional-skills"},

    # --- More project experience ---
    {"id": "project-langgraph-agent", "text": "请介绍你使用 LangGraph 构建 AI Agent 的项目：状态机设计、工具集成和人机协作的实现", "skill_areas": ["LangGraph", "AI agent", "project experience"], "round_type": "project-experience"},
    {"id": "project-vector-search", "text": "分享你实现向量检索系统的经验：索引构建、ANN 搜索优化和 QPS 调优的过程", "skill_areas": ["vector search", "ANN", "project experience"], "round_type": "project-experience"},
    {"id": "project-cicd-pipeline", "text": "Describe your CI/CD pipeline optimization project: from 30-minute builds to 5-minute incremental pipelines.", "skill_areas": ["CI/CD", "optimization", "project experience"], "round_type": "project-experience"},
    {"id": "project-observability-platform", "text": "请分享你搭建可观测性平台的经验：从零到一构建 Tracing + Metrics + Logging 的统一方案", "skill_areas": ["observability", "platform", "project experience"], "round_type": "project-experience"},
    {"id": "project-db-optimization", "text": "Describe a database performance crisis you resolved: slow queries, missing indexes, and connection pool exhaustion.", "skill_areas": ["database", "performance", "project experience"], "round_type": "project-experience"},
    {"id": "project-microservice-split", "text": "请分享你将单体应用拆分为微服务的经验：限界上下文识别、数据拆分和渐进式迁移", "skill_areas": ["microservices", "migration", "project experience"], "round_type": "project-experience"},
]


def expand_question_bank() -> int:
    bank_path = DATASETS / "interview_question_bank.json"
    original = json.loads(bank_path.read_text(encoding="utf-8"))
    orig_count = len(original)
    print(f"Original questions: {orig_count}")

    existing_ids = {q["id"] for q in original}
    new_entries = [q for q in NEW_QUESTIONS if q["id"] not in existing_ids]
    print(f"New questions to add: {len(new_entries)}")

    combined = original + new_entries
    bank_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Total questions: {len(combined)}")
    return len(combined)


# ---------------------------------------------------------------------------
# New eval cases (~90 new, total ~120)
# ---------------------------------------------------------------------------

NEW_CASES = [
    # --- New Go queries ---
    {"case_id": "embed-eval-go-concurrency", "query": "Go goroutine 和 channel 怎么协作？如何避免 goroutine 泄漏？", "round_type": "professional-skills", "expected_question_ids": ["q-go-goroutine", "q-go-context"], "acceptable_skill_areas": ["Go", "concurrency", "backend"], "negative_question_ids": ["q-vue-composables", "q-css-responsive-grid"]},
    {"case_id": "embed-eval-go-en", "query": "Explain Go's escape analysis and garbage collection impact on performance.", "round_type": "professional-skills", "expected_question_ids": ["q-go-memory"], "acceptable_skill_areas": ["Go", "memory", "GC"], "negative_question_ids": ["q-agile-scrum"]},

    # --- New Rust queries ---
    {"case_id": "embed-eval-rust-ownership", "query": "Rust 的所有权和借用规则是什么？lifetime 怎么标注？", "round_type": "professional-skills", "expected_question_ids": ["q-rust-ownership"], "acceptable_skill_areas": ["Rust", "memory safety"], "negative_question_ids": ["q-vue-reactivity"]},
    {"case_id": "embed-eval-rust-async", "query": "Rust tokio async runtime vs Go goroutine: 并发模型的区别？", "round_type": "professional-skills", "expected_question_ids": ["q-rust-async", "q-go-goroutine"], "acceptable_skill_areas": ["Rust", "Go", "concurrency"], "negative_question_ids": ["q-css-responsive-grid"]},

    # --- New K8s queries ---
    {"case_id": "embed-eval-k8s-arch", "query": "Kubernetes 核心组件 API Server、Scheduler、etcd 怎么协作调度一个 Pod？", "round_type": "professional-skills", "expected_question_ids": ["q-k8s-architecture", "q-k8s-pod-lifecycle"], "acceptable_skill_areas": ["Kubernetes", "orchestration"], "negative_question_ids": ["q-vue-composables"]},
    {"case_id": "embed-eval-k8s-network", "query": "Kubernetes CNI and Service networking: how does ClusterIP, NodePort, LoadBalancer differ?", "round_type": "professional-skills", "expected_question_ids": ["q-k8s-networking"], "acceptable_skill_areas": ["Kubernetes", "networking"], "negative_question_ids": ["q-agile-scrum"]},
    {"case_id": "embed-eval-k8s-mixed", "query": "Helm Chart 的最佳实践和 K8s Operator 模式：when to build an Operator instead of Helm?", "round_type": "professional-skills", "expected_question_ids": ["q-k8s-helm", "q-k8s-operator"], "acceptable_skill_areas": ["Kubernetes", "Helm", "Operator"], "negative_question_ids": ["q-vue-reactivity"]},

    # --- New React queries ---
    {"case_id": "embed-eval-react-hooks", "query": "React Hooks 的闭包陷阱是怎么回事？useEffect 依赖数组怎么正确写？", "round_type": "professional-skills", "expected_question_ids": ["q-react-hooks"], "acceptable_skill_areas": ["React", "Hooks"], "negative_question_ids": ["q-database-indexing"]},
    {"case_id": "embed-eval-react-state-en", "query": "Compare Redux Toolkit, Zustand, and React Query for state management in a large React app.", "round_type": "professional-skills", "expected_question_ids": ["q-react-state"], "acceptable_skill_areas": ["React", "state management"], "negative_question_ids": ["q-agile-scrum"]},
    {"case_id": "embed-eval-react-perf", "query": "React 性能优化：React.memo、useMemo、虚拟列表怎么实际落地？", "round_type": "professional-skills", "expected_question_ids": ["q-react-rendering"], "acceptable_skill_areas": ["React", "performance"], "negative_question_ids": ["q-vue-sse-streaming"]},

    # --- New ML queries ---
    {"case_id": "embed-eval-ml-transformer", "query": "Transformer Self-Attention 的 Q、K、V 计算过程和 Multi-Head Attention 原理", "round_type": "professional-skills", "expected_question_ids": ["q-ml-transformer", "q-llm-structured-output"], "acceptable_skill_areas": ["machine learning", "Transformer", "NLP"], "negative_question_ids": ["q-vue-reactivity", "q-frontend-css-layout"]},
    {"case_id": "embed-eval-ml-finetune", "query": "LoRA QLoRA P-Tuning 大模型微调方法的原理和区别", "round_type": "professional-skills", "expected_question_ids": ["q-ml-finetune"], "acceptable_skill_areas": ["machine learning", "fine-tuning"], "negative_question_ids": ["q-agile-scrum"]},
    {"case_id": "embed-eval-ml-rag-vs-ft", "query": "RAG vs Fine-tuning for LLM applications: cost, latency, and accuracy comparison", "round_type": "professional-skills", "expected_question_ids": ["q-ml-rag-vs-finetune", "q-rag-retrieval-pipeline"], "acceptable_skill_areas": ["machine learning", "RAG", "fine-tuning"], "negative_question_ids": ["q-vue-composables"]},
    {"case_id": "embed-eval-ml-vectordb", "query": "向量数据库 Milvus Qdrant Pinecone Weaviate 的选型对比", "round_type": "professional-skills", "expected_question_ids": ["q-ml-vector-db", "q-milvus-vector-search"], "acceptable_skill_areas": ["vector database", "Milvus"], "negative_question_ids": ["q-css-responsive-grid"]},

    # --- New Data Engineering queries ---
    {"case_id": "embed-eval-de-kafka", "query": "Kafka partition consumer group offset 机制如何保证消息可靠性和顺序？", "round_type": "professional-skills", "expected_question_ids": ["q-de-kafka", "q-message-queue"], "acceptable_skill_areas": ["Kafka", "streaming", "message queue"], "negative_question_ids": ["q-vue-reactivity"]},
    {"case_id": "embed-eval-de-spark-en", "query": "Apache Spark RDD vs DataFrame vs Dataset: when to use which and performance implications?", "round_type": "professional-skills", "expected_question_ids": ["q-de-spark"], "acceptable_skill_areas": ["Spark", "big data"], "negative_question_ids": ["q-frontend-css-layout"]},

    # --- New Network queries ---
    {"case_id": "embed-eval-tcp", "query": "TCP 三次握手四次挥手的过程？为什么需要 TIME_WAIT 状态和 2MSL？", "round_type": "professional-skills", "expected_question_ids": ["q-tcp-ip"], "acceptable_skill_areas": ["network", "TCP"], "negative_question_ids": ["q-vue-composables"]},
    {"case_id": "embed-eval-tls", "query": "TLS 1.3 handshake vs TLS 1.2: what changed and what are the 0-RTT security risks?", "round_type": "professional-skills", "expected_question_ids": ["q-tls-handshake"], "acceptable_skill_areas": ["network", "TLS", "security"], "negative_question_ids": ["q-agile-scrum"]},

    # --- New Algorithms queries ---
    {"case_id": "embed-eval-algo-hash", "query": "哈希表的开放寻址和链地址法有什么区别？工业级哈希表怎么设计？", "round_type": "professional-skills", "expected_question_ids": ["q-algo-hash"], "acceptable_skill_areas": ["algorithms", "hash table"], "negative_question_ids": ["q-vue-sse-streaming"]},
    {"case_id": "embed-eval-algo-consensus", "query": "分布式一致性算法 Paxos 和 Raft 的原理对比", "round_type": "professional-skills", "expected_question_ids": ["q-algo-consensus"], "acceptable_skill_areas": ["algorithms", "distributed", "consensus"], "negative_question_ids": ["q-frontend-css-layout"]},

    # --- New Linux/OS queries ---
    {"case_id": "embed-eval-linux-io", "query": "Linux I/O 多路复用：select poll epoll io_uring 的演进和性能差异", "round_type": "professional-skills", "expected_question_ids": ["q-linux-io", "q-python-async-await"], "acceptable_skill_areas": ["Linux", "I/O", "performance"], "negative_question_ids": ["q-vue-reactivity"]},

    # --- New DB queries ---
    {"case_id": "embed-eval-db-sharding", "query": "数据库分库分表：垂直拆分和水平拆分的区别？跨分片查询怎么解决？", "round_type": "professional-skills", "expected_question_ids": ["q-database-sharding"], "acceptable_skill_areas": ["database", "sharding"], "negative_question_ids": ["q-vue-composables"]},
    {"case_id": "embed-eval-db-postgres", "query": "PostgreSQL vs MySQL MVCC implementation differences and index type comparison", "round_type": "professional-skills", "expected_question_ids": ["q-postgres-vs-mysql"], "acceptable_skill_areas": ["PostgreSQL", "MySQL", "database"], "negative_question_ids": ["q-agile-scrum"]},

    # --- New Security queries ---
    {"case_id": "embed-eval-owasp", "query": "OWASP Top 10 安全风险在实际项目中的防护怎么做？", "round_type": "professional-skills", "expected_question_ids": ["q-owasp-top10", "q-security-api-auth"], "acceptable_skill_areas": ["security", "OWASP"], "negative_question_ids": ["q-css-responsive-grid"]},
    {"case_id": "embed-eval-cors", "query": "CORS CSRF SameSite Cookie 的安全防护原理", "round_type": "professional-skills", "expected_question_ids": ["q-cors-csrf"], "acceptable_skill_areas": ["security", "CORS"], "negative_question_ids": ["q-vue-sse-streaming"]},

    # --- New Architecture queries ---
    {"case_id": "embed-eval-cqrs-es", "query": "CQRS Event Sourcing 架构中事件存储和投影怎么实现？最终一致性怎么处理？", "round_type": "professional-skills", "expected_question_ids": ["q-cqrs-event-sourcing", "q-event-driven"], "acceptable_skill_areas": ["CQRS", "Event Sourcing"], "negative_question_ids": ["q-frontend-css-layout"]},
    {"case_id": "embed-eval-service-mesh", "query": "Service Mesh Istio 的流量管理、mTLS 和可观测性是怎么实现的？", "round_type": "professional-skills", "expected_question_ids": ["q-service-mesh"], "acceptable_skill_areas": ["Service Mesh", "Istio"], "negative_question_ids": ["q-agile-scrum"]},

    # --- New TypeScript queries ---
    {"case_id": "embed-eval-ts-advanced", "query": "TypeScript infer extends 条件类型和模板字面量类型的高级应用", "round_type": "professional-skills", "expected_question_ids": ["q-ts-advanced-types", "q-typescript-generics"], "acceptable_skill_areas": ["TypeScript", "types"], "negative_question_ids": ["q-database-indexing"]},

    # --- New DevOps queries ---
    {"case_id": "embed-eval-terraform", "query": "Terraform state management workspace module 设计的最佳实践", "round_type": "professional-skills", "expected_question_ids": ["q-terraform"], "acceptable_skill_areas": ["Terraform", "IaC"], "negative_question_ids": ["q-vue-composables"]},
    {"case_id": "embed-eval-deploy-strategy", "query": "蓝绿部署金丝雀发布在 Kubernetes 中怎么实现零停机？", "round_type": "professional-skills", "expected_question_ids": ["q-blue-green", "q-k8s-pod-lifecycle"], "acceptable_skill_areas": ["DevOps", "Kubernetes", "deployment"], "negative_question_ids": ["q-vue-reactivity"]},

    # --- New Query Types ---
    {"case_id": "embed-eval-multi-domains", "query": "一个全栈工程师需要掌握哪些技能？从 Go Python TypeScript React 到 K8s Terraform", "round_type": "professional-skills", "expected_question_ids": ["q-go-goroutine", "q-react-hooks", "q-k8s-architecture", "q-terraform"], "acceptable_skill_areas": ["Go", "React", "Kubernetes", "Terraform"], "negative_question_ids": ["q-agile-scrum", "q-css-responsive-grid"]},
    {"case_id": "embed-eval-code-switch-4", "query": "K8s 的 Pod lifecycle and liveness/readiness probes 怎么 design for 高可用 distributed systems?", "round_type": "professional-skills", "expected_question_ids": ["q-k8s-pod-lifecycle"], "acceptable_skill_areas": ["Kubernetes", "distributed"], "negative_question_ids": ["q-vue-reactivity"]},
    {"case_id": "embed-eval-code-switch-5", "query": "Is Rust's ownership 系统 suitable for 构建高性能的 Web后端 services compared to Go?", "round_type": "professional-skills", "expected_question_ids": ["q-rust-ownership", "q-go-vs-python"], "acceptable_skill_areas": ["Rust", "Go", "backend"], "negative_question_ids": ["q-css-responsive-grid"]},

    # --- More Chinese heavy queries ---
    {"case_id": "embed-eval-cn-heavy-1", "query": "技术债务怎么管理和推动偿还？如何量化技术债务的严重程度？", "round_type": "professional-skills", "expected_question_ids": ["q-tech-debt"], "acceptable_skill_areas": ["tech debt", "engineering management"], "negative_question_ids": ["q-vue-sse-streaming"]},
    {"case_id": "embed-eval-cn-heavy-2", "query": "跨团队技术项目的推动：资源冲突和优先级分歧怎么处理？", "round_type": "professional-skills", "expected_question_ids": ["q-cross-team"], "acceptable_skill_areas": ["collaboration", "leadership"], "negative_question_ids": ["q-database-indexing"]},
    {"case_id": "embed-eval-cn-heavy-3", "query": "作为 Tech Lead 怎么做技术选型和决策？理想主义和业务交付怎么平衡？", "round_type": "professional-skills", "expected_question_ids": ["q-tech-lead"], "acceptable_skill_areas": ["tech leadership", "decision making"], "negative_question_ids": ["q-frontend-css-layout"]},

    # --- More short keyword queries ---
    {"case_id": "embed-eval-short-k8s", "query": "Kubernetes pod networking", "round_type": "professional-skills", "expected_question_ids": ["q-k8s-networking", "q-k8s-architecture"], "acceptable_skill_areas": ["Kubernetes", "networking"], "negative_question_ids": ["q-vue-composables"]},
    {"case_id": "embed-eval-short-ml", "query": "Transformer attention", "round_type": "professional-skills", "expected_question_ids": ["q-ml-transformer"], "acceptable_skill_areas": ["Transformer", "machine learning"], "negative_question_ids": ["q-agile-scrum"]},
    {"case_id": "embed-eval-short-go", "query": "goroutine channel", "round_type": "professional-skills", "expected_question_ids": ["q-go-goroutine"], "acceptable_skill_areas": ["Go", "concurrency"], "negative_question_ids": ["q-css-responsive-grid"]},

    # --- More project experience queries ---
    {"case_id": "embed-eval-project-k8s", "query": "请分享 Kubernetes 集群迁移的项目经验：有状态服务、网络和监控的挑战", "round_type": "project-experience", "expected_question_ids": ["project-k8s-migration", "q-k8s-architecture"], "acceptable_skill_areas": ["Kubernetes", "migration", "project experience"], "negative_question_ids": ["q-vue-composables", "q-frontend-css-layout"]},
    {"case_id": "embed-eval-project-ml", "query": "Describe your ML pipeline experience: data collection, feature engineering, model training, and online inference.", "round_type": "project-experience", "expected_question_ids": ["project-ml-pipeline"], "acceptable_skill_areas": ["machine learning", "pipeline", "project experience"], "negative_question_ids": ["q-vue-reactivity"]},

    # --- More mixed CN/EN ---
    {"case_id": "embed-eval-mixed-react", "query": "React 的 Virtual DOM diff 算法 is 怎么 working 的？和 Vue 的响应式系统有何不同？", "round_type": "professional-skills", "expected_question_ids": ["q-react-rendering", "q-vue-reactivity"], "acceptable_skill_areas": ["React", "Vue", "frontend"], "negative_question_ids": ["q-database-indexing"]},
    {"case_id": "embed-eval-mixed-k8s", "query": "K8s 的 Service Mesh Istio 怎么实现 traffic splitting and 金丝雀发布 canary deployment?", "round_type": "professional-skills", "expected_question_ids": ["q-service-mesh", "q-blue-green"], "acceptable_skill_areas": ["Service Mesh", "Kubernetes", "DevOps"], "negative_question_ids": ["q-agile-scrum"]},
    {"case_id": "embed-eval-mixed-ml", "query": "LLM 的 fine-tuning methods like LoRA and QLoRA 的原理是什么？和 full fine-tuning比 有什么 trade-off?", "round_type": "professional-skills", "expected_question_ids": ["q-ml-finetune"], "acceptable_skill_areas": ["machine learning", "fine-tuning", "LLM"], "negative_question_ids": ["q-css-responsive-grid"]},

    # --- Long description queries ---
    {"case_id": "embed-eval-long-ml", "query": "我正在构建一个基于 LLM 的智能客服系统，需要选择合适的 embedding 模型、向量数据库和 RAG pipeline 架构。系统要支持中英文混合查询、高并发低延迟，并保证检索质量。请问在技术选型和架构设计上有哪些关键考虑？", "round_type": "professional-skills", "expected_question_ids": ["q-ml-vector-db", "q-rag-retrieval-pipeline", "q-ml-rag-vs-finetune", "q-rag-embedding-selection"], "acceptable_skill_areas": ["RAG", "vector database", "embedding", "LLM"], "negative_question_ids": ["q-css-responsive-grid", "q-vue-reactivity"]},
    {"case_id": "embed-eval-long-platform", "query": "Design a cloud-native interview platform that uses Go microservices, React frontend, Kubernetes orchestration, LangGraph for AI interview state machine, Milvus for question retrieval, and Kafka for async report generation. What are the key architectural decisions?", "round_type": "professional-skills", "expected_question_ids": ["q-system-design-interview-system", "q-go-microservice-saga", "q-k8s-architecture", "q-de-kafka"], "acceptable_skill_areas": ["system design", "cloud native", "microservices"], "negative_question_ids": ["q-agile-scrum", "q-frontend-css-layout"]},

    # --- Observability depth queries ---
    {"case_id": "embed-eval-o11y-metrics", "query": "RED 和 USE 方法论在微服务监控中怎么应用？什么指标需要设置告警？", "round_type": "professional-skills", "expected_question_ids": ["q-metrics-design", "q-alerting-strategy", "q-otel-tracing"], "acceptable_skill_areas": ["observability", "metrics", "SRE"], "negative_question_ids": ["q-vue-reactivity"]},

    # --- Additional diverse queries ---
    {"case_id": "embed-eval-software-eng", "query": "TDD 在实际项目中真的可行吗？代码重构的策略和时机怎么把握？", "round_type": "professional-skills", "expected_question_ids": ["q-tdd-practice", "q-refactoring"], "acceptable_skill_areas": ["TDD", "refactoring"], "negative_question_ids": ["q-frontend-css-layout"]},
    {"case_id": "embed-eval-sre", "query": "On-call incident response 流程和 postmortem 文化怎么建立？", "round_type": "professional-skills", "expected_question_ids": ["q-oncall"], "acceptable_skill_areas": ["SRE", "incident response"], "negative_question_ids": ["q-vue-composables"]},
    {"case_id": "embed-eval-chaos", "query": "混沌工程在生产环境中的故障注入怎么做？如何保证安全？", "round_type": "professional-skills", "expected_question_ids": ["q-chaos-engineering"], "acceptable_skill_areas": ["Chaos Engineering", "SRE"], "negative_question_ids": ["q-agile-scrum"]},
    {"case_id": "embed-eval-mentoring", "query": "How to mentor junior engineers and build effective code review culture?", "round_type": "professional-skills", "expected_question_ids": ["q-mentoring", "q-code-review-practice"], "acceptable_skill_areas": ["mentoring", "engineering culture"], "negative_question_ids": ["q-css-responsive-grid"]},
    {"case_id": "embed-eval-hexagonal", "query": "六边形架构在 Python 项目中怎么实现？端口和适配器如何解耦业务逻辑？", "round_type": "professional-skills", "expected_question_ids": ["q-hexagonal-arch"], "acceptable_skill_areas": ["hexagonal architecture", "Python"], "negative_question_ids": ["q-vue-sse-streaming"]},

    # ===== BATCH 2: New eval cases =====
    {"case_id": "embed-eval-graphql-n1", "query": "GraphQL 的 N+1 问题怎么用 DataLoader 解决？批处理和缓存的原理", "round_type": "professional-skills", "expected_question_ids": ["q-graphql-nplusone"], "acceptable_skill_areas": ["GraphQL", "performance"], "negative_question_ids": ["q-vue-reactivity"]},
    {"case_id": "embed-eval-circuit-breaker", "query": "熔断器模式的三种状态和半开探测怎么实现？", "round_type": "professional-skills", "expected_question_ids": ["q-circuit-breaker"], "acceptable_skill_areas": ["circuit breaker", "resilience"], "negative_question_ids": ["q-css-responsive-grid"]},
    {"case_id": "embed-eval-ml-rlhf", "query": "RLHF reward model 训练和 PPO 优化的原理", "round_type": "professional-skills", "expected_question_ids": ["q-ml-rlhf"], "acceptable_skill_areas": ["RLHF", "LLM"], "negative_question_ids": ["q-agile-scrum"]},
    {"case_id": "embed-eval-ml-quant", "query": "GPTQ AWQ GGUF 模型量化的原理和推理性能比较", "round_type": "professional-skills", "expected_question_ids": ["q-ml-quantization"], "acceptable_skill_areas": ["quantization", "LLM"], "negative_question_ids": ["q-vue-composables"]},
    {"case_id": "embed-eval-ml-moe", "query": "MoE 混合专家架构的路由机制和负载均衡怎么实现？", "round_type": "professional-skills", "expected_question_ids": ["q-ml-moe"], "acceptable_skill_areas": ["MoE", "architecture"], "negative_question_ids": ["q-frontend-css-layout"]},
    {"case_id": "embed-eval-agent-framework", "query": "LangChain LangGraph CrewAI AutoGen AI Agent 框架的架构对比", "round_type": "professional-skills", "expected_question_ids": ["q-ml-agent-framework", "q-langgraph-state-machine"], "acceptable_skill_areas": ["AI agent", "framework"], "negative_question_ids": ["q-css-responsive-grid"]},
    {"case_id": "embed-eval-istio-ambient", "query": "Istio Sidecar vs Ambient Mesh: 无 Sidecar 服务网格的优势和原理", "round_type": "professional-skills", "expected_question_ids": ["q-istio-deep", "q-service-mesh"], "acceptable_skill_areas": ["Istio", "Service Mesh"], "negative_question_ids": ["q-vue-reactivity"]},
    {"case_id": "embed-eval-e2e-testing", "query": "Playwright Cypress Selenium 端到端测试框架的对比和选型", "round_type": "professional-skills", "expected_question_ids": ["q-e2e-playwright"], "acceptable_skill_areas": ["testing", "e2e"], "negative_question_ids": ["q-database-indexing"]},
    {"case_id": "embed-eval-contract-test", "query": "契约测试 Pact 框架如何在微服务中保证接口兼容性？", "round_type": "professional-skills", "expected_question_ids": ["q-contract-test", "q-testing-pyramid"], "acceptable_skill_areas": ["testing", "contract test"], "negative_question_ids": ["q-agile-scrum"]},
    {"case_id": "embed-eval-gitops", "query": "Argo CD GitOps 工作流的 Application 和自动同步策略", "round_type": "professional-skills", "expected_question_ids": ["q-argo-cd"], "acceptable_skill_areas": ["Argo CD", "GitOps"], "negative_question_ids": ["q-vue-sse-streaming"]},
    {"case_id": "embed-eval-vue-slots", "query": "Vue 作用域插槽和动态插槽名的高级用法", "round_type": "professional-skills", "expected_question_ids": ["q-vue-slots", "q-vue-component-communication"], "acceptable_skill_areas": ["Vue", "slots"], "negative_question_ids": ["q-database-indexing"]},
    {"case_id": "embed-eval-react-fiber", "query": "React Fiber 可中断渲染和优先级调度的原理", "round_type": "professional-skills", "expected_question_ids": ["q-react-fiber"], "acceptable_skill_areas": ["React", "Fiber"], "negative_question_ids": ["q-agile-scrum"]},
    {"case_id": "embed-eval-wasm", "query": "WebAssembly 在前端图像处理和加密计算中的应用场景", "round_type": "professional-skills", "expected_question_ids": ["q-wasm-frontend"], "acceptable_skill_areas": ["WebAssembly", "frontend"], "negative_question_ids": ["q-css-responsive-grid"]},
    {"case_id": "embed-eval-backpressure", "query": "Reactive Streams 背压机制在流式系统中的实现原理", "round_type": "professional-skills", "expected_question_ids": ["q-backpressure"], "acceptable_skill_areas": ["backpressure", "streaming"], "negative_question_ids": ["q-vue-reactivity"]},
    {"case_id": "embed-eval-project-agent", "query": "请分享你用 LangGraph 构建 AI Agent 的项目经验：状态机、工具和人机协作", "round_type": "project-experience", "expected_question_ids": ["project-langgraph-agent", "q-langgraph-human-loop"], "acceptable_skill_areas": ["LangGraph", "AI agent", "project experience"], "negative_question_ids": ["q-css-responsive-grid", "q-frontend-css-layout"]},
    {"case_id": "embed-eval-project-o11y", "query": "Describe your experience building an observability platform: unified Tracing + Metrics + Logging.", "round_type": "project-experience", "expected_question_ids": ["project-observability-platform", "q-otel-tracing", "q-metrics-design"], "acceptable_skill_areas": ["observability", "platform", "project experience"], "negative_question_ids": ["q-vue-composables"]},
    {"case_id": "embed-eval-rust-error", "query": "Rust Result 和 ? 操作符的错误处理机制 anyhow vs thiserror", "round_type": "professional-skills", "expected_question_ids": ["q-rust-error"], "acceptable_skill_areas": ["Rust", "error handling"], "negative_question_ids": ["q-agile-scrum"]},
    {"case_id": "embed-eval-frontend-bundler", "query": "Webpack Vite Turbopack esbuild 前端构建工具的实现原理对比", "round_type": "professional-skills", "expected_question_ids": ["q-frontend-bundler"], "acceptable_skill_areas": ["frontend", "bundler"], "negative_question_ids": ["q-database-indexing"]},
]


def expand_eval_cases() -> int:
    cases_path = DATASETS / "embedding_eval_cases.json"
    original = json.loads(cases_path.read_text(encoding="utf-8"))
    orig_count = len(original)

    existing_ids = {c["case_id"] for c in original}
    new_entries = [c for c in NEW_CASES if c["case_id"] not in existing_ids]

    combined = original + new_entries
    cases_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Original cases: {orig_count}, New: {len(new_entries)}, Total: {len(combined)}")
    return len(combined)


if __name__ == "__main__":
    bank_count = expand_question_bank()
    case_count = expand_eval_cases()
    print(f"\nDone! Questions: {bank_count}, Eval cases: {case_count}")
