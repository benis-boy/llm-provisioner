# Video Recording

Capture browser automation sessions as video for debugging, documentation, or verification. Produces WebM output.

## Basic Recording

```bash
playwright-cli open
playwright-cli video-start demo.webm
playwright-cli video-chapter "Getting Started" --description="Opening the homepage" --duration=2000
playwright-cli goto https://example.com
playwright-cli snapshot
playwright-cli click e1
playwright-cli video-chapter "Filling Form" --description="Entering test data" --duration=2000
playwright-cli fill e2 "test input"
playwright-cli video-stop
```

## Best Practices

### 1. Use Descriptive Filenames

```bash
playwright-cli video-start recordings/login-flow.webm
playwright-cli video-start recordings/checkout-test-run-42.webm
```

### 2. Record entire hero scripts

When recording a video for the user or as proof of work, create a code snippet and execute it with `run-code` so you can add pauses and overlays.

```js
async page => {
  await page.screencast.start({ path: 'video.webm', size: { width: 1280, height: 800 } });
  await page.goto('https://demo.playwright.dev/todomvc');
  await page.screencast.showChapter('Adding Todo Items', {
    description: 'We will add several items to the todo list.',
    duration: 2000,
  });
  await page.getByRole('textbox', { name: 'What needs to be done?' }).pressSequentially('Walk the dog', { delay: 60 });
  await page.getByRole('textbox', { name: 'What needs to be done?' }).press('Enter');
  await page.waitForTimeout(1000);
  await page.screencast.stop();
}
```

## Overlay API Summary

| Method | Use Case |
|--------|----------|
| `page.screencast.showChapter(...)` | Full-screen chapter card |
| `page.screencast.showOverlay(...)` | Custom HTML overlay |
| `disposable.dispose()` | Remove a sticky overlay |

## Tracing vs Video

| Feature | Video | Tracing |
|---------|-------|---------|
| Use case | Demos, documentation | Debugging, analysis |

## Limitations

Record only the named scenario and use disposable output paths. Videos can
contain secrets and user data; do not commit them or claim browser readiness
without a real smoke check in the target environment.

- Recording adds slight overhead to automation
- Large recordings can consume significant disk space
