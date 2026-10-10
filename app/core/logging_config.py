"""
Production Structured Logging Configuration for Google Cloud Run and MLOps monitoring.
"""

import datetime
import json
import logging
import sys
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """
    JSON Formatter for Google Cloud Logging compatibility.
    Converts log records to JSON structured objects parsed automatically by GCP Cloud Logging.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_payload: Dict[str, Any] = {
            "timestamp": datetime.datetime.fromtimestamp(
                record.created, tz=datetime.timezone.utc
            ).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Include exception details if present
        if record.exc_info:
            log_payload["exception"] = self.formatException(record.exc_info)

        # Include custom extra metadata passed to logger
        if hasattr(record, "extra_payload") and isinstance(record.extra_payload, dict):
            log_payload.update(record.extra_payload)

        return json.dumps(log_payload)


class StructuredTextFormatter(logging.Formatter):
    """
    Clean, human-readable structured text formatter for development and standard log streaming.
    Format: [TIMESTAMP] [SEVERITY] [LOGGER:LINE] MESSAGE
    """

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.datetime.fromtimestamp(
            record.created, tz=datetime.timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + "Z"
        level = f"{record.levelname:<7}"
        location = f"[{record.name}:{record.lineno}]"
        message = record.getMessage()

        formatted = f"{timestamp} | {level} | {location:<30} | {message}"

        if record.exc_info:
            formatted += "\n" + self.formatException(record.exc_info)

        return formatted


def configure_logging(
    log_level: str = "INFO", json_format: bool = False
) -> logging.Logger:
    """
    Configures the root and application loggers with structured formatting.

    Args:
        log_level: Minimum logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        json_format: If True, uses JSONFormatter (ideal for Google Cloud Logging).
                     If False, uses StructuredTextFormatter (ideal for standard streams).

    Returns:
        The configured application root logger.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplication
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    # Output to stdout for Cloud Run log capture
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(numeric_level)

    if json_format:
        stream_handler.setFormatter(JSONFormatter())
    else:
        stream_handler.setFormatter(StructuredTextFormatter())

    root_logger.addHandler(stream_handler)

    # Silence overly verbose third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("comet_ml").setLevel(logging.INFO)

    logger = logging.getLogger("fraud_api")
    logger.info(
        f"Structured logging initialized (level={log_level}, json_format={json_format})"
    )
    return logger
