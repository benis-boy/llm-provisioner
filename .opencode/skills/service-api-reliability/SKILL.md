---
name: service-api-reliability
description: Use when designing, implementing, reviewing, or documenting service-to-service APIs, retries, error responses, OpenAPI contracts, or microservice failure handling.
---

# Service API Reliability

Use this skill for any backend or service work that involves:

- HTTP APIs
- service-to-service calls
- retry behavior
- error payloads
- OpenAPI contract design
- failure handling across microservices

## Reliability Rules

- Treat every service as independently failing.
- Any service may be unavailable, slow, or transiently failing at any time.
- All service-to-service API calls must support retries with exponential backoff.
- Default retry policy: retry up to 5 times before failing permanently.
- If an error is marked non-retryable, callers must stop retrying immediately.

## Shared API Error Contract

All APIs must return the same structured error payload shape.

Use lowerCamelCase field names in API payloads.

Canonical response shape:

```json
{
  "error": {
    "message": "human-readable summary",
    "errorCode": 409001,
    "retryable": false,
    "context": {
      "publicationId": "8a3c...",
      "filename": "chapter-01.md"
    },
    "causedBy": {
      "message": "optional nested cause",
      "errorCode": 404001,
      "retryable": false,
      "context": {}
    },
    "stackTrace": "optional debug-only stack trace",
    "service": "api",
    "requestId": "req_123",
    "timestamp": "2026-08-02T12:34:56Z"
  }
}
```

## Field Requirements

- `message`: required string for humans
- `errorCode`: required 6-digit integer for machine handling and translating
- `retryable`: required boolean used by callers to decide whether to retry
- `context`: required object for structured key-value details
- `causedBy`: optional nested error object for a direct underlying cause
- `stackTrace`: optional string
- `service`: required string naming the responding service
- `requestId`: required string for tracing
- `timestamp`: required RFC 3339 timestamp string

Additional rules:

- `context` values may be strings, numbers, booleans, arrays, or objects as needed for machine-readable diagnostics.
- `causedBy` should be limited to a short direct cause chain and must not grow without bound.
- Error payloads must not leak secrets, credentials, tokens, or raw sensitive document contents.

## Error Code Rules

- Error codes must be 6 digits.
- The first 3 digits must be the HTTP status code.
- The last 3 digits identify the application-specific error within that HTTP status.

## OpenAPI Guidance

- Reuse the shared error schema in every API spec rather than redefining ad hoc error payloads.

## Implementation Guidance

- Preserve the original cause when wrapping lower-level failures.
- Strip long error-chains down to the essential information only.
- Convert internal errors into the shared public error payload before returning responses.
- Map retryability based on behavior, not just HTTP status.
- Constraint violations and business-rule rejections should generally be non-retryable.
- Temporary network, storage, dependency, and timeout failures should generally be retryable.
- Prefer explicit error codes over parsing error messages.
