from __future__ import annotations

import logging
import os
from collections.abc import Mapping

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import (
    ALWAYS_OFF,
    ALWAYS_ON,
    ParentBased,
    Sampler,
    TraceIdRatioBased,
)

DEFAULT_SERVICE_NAME = "interview-python-agent"
DEFAULT_ENVIRONMENT = "local"
DEFAULT_TRACES_SAMPLER = "parentbased_always_on"
DEFAULT_TRACES_SAMPLER_ARG = 1.0
logger = logging.getLogger(__name__)

_tracer_provider: TracerProvider | None = None
_instrumented_app_ids: set[int] = set()


def interview_protocol_from_message(raw_user_message: str) -> str:
    """Classify the stream protocol without exposing the message content."""
    stripped_message = raw_user_message.lstrip()
    if not stripped_message:
        return "empty"
    if stripped_message.startswith("{"):
        return "structured-start-v1"
    return "reply"


def _sdk_disabled() -> bool:
    return os.getenv("OTEL_SDK_DISABLED", "").lower() == "true"


def _parse_resource_attributes(value: str | None) -> dict[str, str]:
    if not value:
        return {}

    attributes: dict[str, str] = {}
    for entry in value.split(","):
        key, separator, raw_attribute_value = entry.partition("=")
        attribute_key = key.strip()
        attribute_value = raw_attribute_value.strip()
        if separator and attribute_key and attribute_value:
            attributes[attribute_key] = attribute_value
    return attributes


def _build_resource_attributes() -> Mapping[str, str]:
    environment_attributes = _parse_resource_attributes(os.getenv("OTEL_RESOURCE_ATTRIBUTES"))
    service_name = os.getenv("OTEL_SERVICE_NAME") or DEFAULT_SERVICE_NAME
    deployment_environment = (
        environment_attributes.get("deployment.environment")
        or os.getenv("APP_ENV")
        or DEFAULT_ENVIRONMENT
    )

    return {
        "service.name": service_name,
        "deployment.environment": deployment_environment,
        **environment_attributes,
    }


def _get_tracer_provider() -> TracerProvider | None:
    global _tracer_provider

    if _sdk_disabled():
        return None

    if _tracer_provider is not None:
        return _tracer_provider

    provider = TracerProvider(
        resource=Resource.create(_build_resource_attributes()),
        sampler=_build_sampler(),
    )
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    _tracer_provider = provider
    return provider


def _build_sampler() -> Sampler:
    sampler_name = (os.getenv("OTEL_TRACES_SAMPLER") or DEFAULT_TRACES_SAMPLER).lower()

    if sampler_name == "always_on":
        return ALWAYS_ON
    if sampler_name == "always_off":
        return ALWAYS_OFF
    if sampler_name == "parentbased_always_on":
        return ParentBased(ALWAYS_ON)
    if sampler_name == "parentbased_always_off":
        return ParentBased(ALWAYS_OFF)
    if sampler_name == "traceidratio":
        return TraceIdRatioBased(_sampler_ratio_from_env())
    if sampler_name == "parentbased_traceidratio":
        return ParentBased(TraceIdRatioBased(_sampler_ratio_from_env()))

    logger.warning(
        "Unsupported OTEL_TRACES_SAMPLER=%s; defaulting to %s.",
        sampler_name,
        DEFAULT_TRACES_SAMPLER,
    )
    return ParentBased(ALWAYS_ON)


def _sampler_ratio_from_env() -> float:
    raw_value = os.getenv("OTEL_TRACES_SAMPLER_ARG")
    if raw_value is None or raw_value.strip() == "":
        return DEFAULT_TRACES_SAMPLER_ARG

    try:
        value = float(raw_value)
    except ValueError:
        logger.warning(
            "Invalid OTEL_TRACES_SAMPLER_ARG=%s; defaulting to %.1f.",
            raw_value,
            DEFAULT_TRACES_SAMPLER_ARG,
        )
        return DEFAULT_TRACES_SAMPLER_ARG

    if 0 <= value <= 1:
        return value

    logger.warning(
        "OTEL_TRACES_SAMPLER_ARG=%s is outside [0, 1]; defaulting to %.1f.",
        raw_value,
        DEFAULT_TRACES_SAMPLER_ARG,
    )
    return DEFAULT_TRACES_SAMPLER_ARG


def instrument_fastapi(app: FastAPI) -> None:
    """Attach OpenTelemetry FastAPI instrumentation to an application."""
    provider = _get_tracer_provider()
    if provider is None:
        return

    app_id = id(app)
    if app_id in _instrumented_app_ids:
        return

    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
    _instrumented_app_ids.add(app_id)
