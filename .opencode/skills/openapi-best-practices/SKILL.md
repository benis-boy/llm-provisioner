---
name: openapi-best-practices
description: OpenAPI, OAS, swagger, openapi.yaml, openapi.yml, openapi.json, paths, components, schemas, $ref. Use when designing, editing, reviewing, or organizing OpenAPI descriptions and HTTP API contracts.
---

# OpenAPI Best Practices

Use this skill when the task involves OpenAPI documents, API-first contract design, or review of an HTTP API description.

## Source Priorities

- Treat the OpenAPI specification at `spec.openapis.org` as normative.
- Prefer the OpenAPI Initiative learning docs for authoring guidance and document organization.
- Follow the repo's existing contract style before introducing new patterns.

## Default OpenAPI Rules

- Prefer design-first when the API surface is still being defined.
- Keep one source of truth for the contract.
- Put reusable schemas, parameters, responses, and security schemes under `components`.
- Use `$ref` instead of repeating equivalent structures.
- Keep examples valid and representative.
- Use consistent operation naming and tag related endpoints.
- Be explicit about error responses and authentication requirements.

## Authoring Guidance

- Start with clear `info`, `servers`, `paths`, and shared `components`.
- Model request and response bodies precisely enough for generators and validators.
- Prefer small, composable schemas over repeated inline objects.
- Split large specs into multiple files when the natural path hierarchy supports it.
- Use relative references carefully and keep the entry document obvious, typically `openapi.yaml` or `openapi.json`.
- Avoid relying on tool-specific or implementation-defined behavior when interoperable alternatives exist.

## Review Checklist

- Does every operation define success and error responses?
- Are repeated structures moved into `components`?
- Are path, query, header, and body inputs modeled in the correct places?
- Are examples aligned with the declared schema?
- Are tags and operation groupings easy to navigate?
- Would client/server generation and validation tools have enough information to behave correctly?

## References

- `https://spec.openapis.org/oas/latest.html`
- `https://learn.openapis.org/best-practices.html`
- `https://learn.openapis.org/specification/components.html`
