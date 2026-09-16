---
name: web-design-guidelines
description: UI, UX, accessibility, responsive design, and interaction review. Use when asked to review or audit frontend design, accessibility, usability, or mobile behavior.
license: MIT
metadata:
  source: vercel-labs/web-interface-guidelines
  adapted-for: book-data-analyzer
---

# Web Design Guidelines

Review the requested UI against the current product design and established styles before proposing visual changes.

## Review Areas

- Semantic structure, labels, keyboard operation, focus visibility, and screen-reader naming.
- Text and control contrast, non-color status cues, reduced-motion behavior, and target sizes.
- Loading, empty, error, disabled, pending, and success states.
- Responsive behavior at narrow mobile widths and wide desktop layouts without horizontal overflow.
- Clear hierarchy, readable line lengths, consistent spacing, and intentional density.
- Form instructions, validation placement, destructive-action confirmation, and recovery paths.
- Stable layouts that avoid unexpected movement and preserve user context during updates.
- Interaction feedback for hover, focus, active, selected, and live-update states.

## Method

1. Inspect the relevant components, styles, and nearby established patterns.
2. Exercise the interface in a browser when behavior or responsiveness cannot be proven statically.
3. Report concrete findings ordered by severity with `file:line` references and the user impact.
4. Distinguish accessibility failures from visual preferences. Avoid style-only findings that do not improve usability or product coherence.
5. Recommend the smallest change that resolves each issue while preserving the current visual language.
