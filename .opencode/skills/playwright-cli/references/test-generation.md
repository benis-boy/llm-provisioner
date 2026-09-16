# Test generation (plan -> generate -> heal)

End-to-end workflow for authoring and maintaining Playwright tests with `playwright-cli`. Every `playwright-cli` action emits the equivalent Playwright TypeScript, and that generated code is the raw material for every test. The sections below can be used independently:

- **How generation works** - the core mechanic everything else relies on: actions become TypeScript, plus how to add assertions.
- **Plan** - explore the app, produce a spec file describing what to test.
- **Generate** - turn a spec into Playwright test files. Update the spec if it's vague or stale.
- **Heal** - diagnose failing tests, fix the code, reconcile the spec with reality.

Plan / generate / heal lean on the same mechanic: run `npx playwright test --debug=cli` in the background, then `playwright-cli attach tw-XXXX` to drive the paused page interactively. See [playwright-tests.md](playwright-tests.md) for the debug/attach mechanics.

---

## 0. How generation works

Every action you perform with `playwright-cli` generates corresponding Playwright TypeScript code. This code appears in the output and can be copied directly into your test files.

```bash
playwright-cli open https://example.com/login
playwright-cli snapshot
playwright-cli fill e1 "user@example.com"
playwright-cli fill e2 "password123"
playwright-cli click e3
```

### Building a test file

Collect the generated code into a Playwright test:

```typescript
import { test, expect } from '@playwright/test';

test('login flow', async ({ page }) => {
  await page.goto('https://example.com/login');
  await page.getByRole('textbox', { name: 'Email' }).fill('user@example.com');
  await page.getByRole('textbox', { name: 'Password' }).fill('password123');
  await page.getByRole('button', { name: 'Sign In' }).click();

  await expect(page).toHaveURL(/.*dashboard/);
});
```

### Use semantic locators

The generated code uses role-based locators when possible, which are more resilient:

```typescript
await page.getByRole('button', { name: 'Submit' }).click();
```

### Explore before recording

Take snapshots to understand the page structure before recording actions:

```bash
playwright-cli open https://example.com
playwright-cli snapshot
playwright-cli click e5
```

### Add assertions manually

Generated code captures actions but not assertions. Add expectations in your test using one of the recommended matchers:

- `toBeVisible()`
- `toHaveText(text)`
- `toHaveValue(value)` / `toBeEmpty()`
- `toBeChecked()` / `toBeUnchecked()`
- `toMatchAriaSnapshot(snapshot)`

Use `playwright-cli generate-locator <target>` to produce the locator expression for the assertion, and the snapshot/eval commands to capture the expected value.

```bash
playwright-cli --raw generate-locator e5
playwright-cli --raw eval "el => el.textContent" e5
playwright-cli --raw snapshot e5
```

---

## 1. Planning

Goal: produce a spec file that enumerates the scenarios to test. Always write the spec to a file.

### 1.1 Prerequisite: workspace

Check the workspace has Playwright installed before anything else:

```bash
test -f playwright.config.ts || test -f playwright.config.js
npx --no-install playwright --version
```

If there is no Playwright install, bootstrap one and let the user pick the defaults:

```bash
npm init playwright@latest
```

### 1.2 Prerequisite: seed test

A seed test is a minimal test that lands the page in the state every scenario starts from.

```ts
import { test } from '@playwright/test';

test('seed', async ({ page }) => {
  await page.goto('https://example.com/');
});
```

### 1.3 Explore the app

Launch the app via the seed in the background and attach:

```bash
PLAYWRIGHT_HTML_OPEN=never npx playwright test tests/seed.spec.ts --debug=cli
playwright-cli attach tw-XXXX
```

Do not just open the app URL with `playwright-cli`; go through the test so any test setup is preserved.

### 1.4 Write the spec file

Save under `specs/<feature>.plan.md`.

```markdown
# <Feature> Test Plan

## Application Overview

<One paragraph describing what the feature does and why it matters.>

## Test Scenarios

### 1. <Group Name>

**Seed:** `tests/seed.spec.ts`

#### 1.1. <kebab-case-scenario-name>

**File:** `tests/<group>/<kebab-case-scenario-name>.spec.ts`
```

---

## 2. Generate

Goal: take a spec file and produce Playwright test files.

### 2.1 Inputs

- Spec file
- Target scenario or group
- Seed file

### 2.2 Generate one scenario

```bash
PLAYWRIGHT_HTML_OPEN=never npx playwright test <seed-file> --debug=cli
playwright-cli attach tw-XXXX
```

Walk the scenario with `playwright-cli`, update the spec if it has drifted, and write one test file per scenario.

```ts
import { test, expect } from '@playwright/test';

test('should sign in', async ({ page }) => {
  await page.getByRole('textbox', { name: 'username' }).fill('John Doe');
  await page.getByRole('textbox', { name: 'password' }).fill('TestPassword');
  await page.getByRole('textbox', { name: 'password' }).press('Enter');
  await expect(page.getByRole('heading')).toContainText('Welcome, John Doe!');
});
```

### 2.3 Generate multiple scenarios

Loop the same workflow one scenario at a time, restarting the seed between each scenario.

### 2.4 Run generated tests

```bash
PLAYWRIGHT_HTML_OPEN=never npx playwright test tests/<group>/<scenario>.spec.ts
```

---

## 3. Heal

Goal: fix failing tests, and update the spec if the app's intended behavior changed.

### 3.1 Find failing tests

```bash
PLAYWRIGHT_HTML_OPEN=never npx playwright test
```

### 3.2 Debug one failure

```bash
PLAYWRIGHT_HTML_OPEN=never npx playwright test tests/<group>/<scenario>.spec.ts:<line> --debug=cli
playwright-cli attach tw-XXXX
```

Use `snapshot`, `console`, `requests`, and `show --annotate` to diagnose the failure.

### 3.3 Apply the fix

Edit the test file, stop the debug run, and rerun the single test to confirm it passes.

### 3.4 Reconcile with the spec

- If the fix is only locator/assertion drift, leave the spec alone.
- If user-visible behavior changed, update the spec.
- If it is unclear whether the app changed intentionally or regressed, ask the user.

### 3.5 Iteration and giving up

- Fix failures one at a time.
- If the test is correct but the app is wrong and the user confirms it is a bug, mark the test `test.fixme(...)` with a comment.

---

## Cross-references

| For... | See |
|---|---|
| `--debug=cli` / attach mechanics | [playwright-tests.md](playwright-tests.md) |
| Mocking requests during exploration/generation | [request-mocking.md](request-mocking.md) |
| Managing the CLI browser session | [session-management.md](session-management.md) |
