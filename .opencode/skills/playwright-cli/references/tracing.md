# Tracing

Capture detailed execution traces for debugging and analysis. Traces include DOM snapshots, screenshots, network activity, and console logs.

## Basic Usage

```bash
playwright-cli tracing-start
playwright-cli open https://example.com
playwright-cli click e1
playwright-cli fill e2 "test"
playwright-cli tracing-stop
```

## Trace Output Files

When you start tracing, Playwright creates a `traces/` directory with several files:

### `trace-{timestamp}.trace`

The main trace file contains actions, DOM snapshots, screenshots, timing information, console messages, and source locations.

### `trace-{timestamp}.network`

The network log contains all HTTP requests and responses with timing and payload details.

### `resources/`

Cached resources needed to reconstruct page state.

## Use Cases

### Debugging Failed Actions

```bash
playwright-cli tracing-start
playwright-cli open https://app.example.com
playwright-cli click e5
playwright-cli tracing-stop
```

### Analyzing Performance

```bash
playwright-cli tracing-start
playwright-cli open https://slow-site.com
playwright-cli tracing-stop
```

### Capturing Evidence

```bash
playwright-cli tracing-start
playwright-cli open https://app.example.com/checkout
playwright-cli fill e1 "4111111111111111"
playwright-cli fill e2 "12/25"
playwright-cli fill e3 "123"
playwright-cli click e4
playwright-cli tracing-stop
```

## Trace vs Video vs Screenshot

| Feature | Trace | Video | Screenshot |
|---------|-------|-------|------------|
| DOM inspection | Yes | No | No |
| Network details | Yes | No | No |
| Step-by-step replay | Yes | Continuous | Single frame |

## Best Practices

### 1. Start Tracing Before the Problem

```bash
playwright-cli tracing-start
playwright-cli open https://example.com
playwright-cli tracing-stop
```

### 2. Clean Up Old Traces

```bash
find .playwright-cli/traces -mtime +7 -delete
```
