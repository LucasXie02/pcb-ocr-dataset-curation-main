# Testing Guide

## Principle

Every code change must pass automated end-to-end tests **before** asking the user to manually verify. Tests simulate real user workflows via headless Chrome (Puppeteer) and **never skip errors**.

Testing is split into two layers:
1. **Per-feature tests** — each API endpoint and UI element works in isolation
2. **Cross-component interaction tests** — actions in one part of the UI correctly propagate to all other parts

The second layer is critical. Most bugs users find are **not** broken APIs — they are stale data, missing cache invalidation, or forgotten state resets after an operation.

## Setup

```bash
npm install puppeteer
npx puppeteer browsers install chrome
```

## Running Tests

```bash
# 1. Start the server
python ocr_review_app.py --dataset_root Data/ocr_zip_files/0311_ocr --port 5001

# 2. Run tests (in another terminal)
node tests/test_e2e.js
```

## Test Structure

### Per-Feature Tests (sections 1–13)

| Section | What it tests |
|---------|--------------|
| Accept line | Load queue → load line data → accept via API |
| Edit line | Change OCR text, modify char bbox, modify line bbox, verify persistence |
| Add line | Add new line with chars → verify loadable |
| Delete line | Delete line → verify gone |
| Rotate image | Single image CW+CCW roundtrip, verify image still serves |
| Accept group | Gallery → get group → accept all siblings |
| Reject instance | Get group detail → reject one instance |
| Batch rotate | Group rotate CW+CCW roundtrip |
| Batch accept agreeing | Auto-accept all high-confidence agreeing groups |
| Dirty state + save | Edit text in canvas → verify dirty → save → verify clean |
| Board switch consistency | Switch A→B→A, verify gallery data matches |
| Gallery UI | Cards, thumbnails, filter, sort, position, keyboard nav, comparison modal |
| Review tab controls | Mode buttons, navigation, sliders |

### Cross-Component Interaction Tests (section 14)

These test that **an action in one place updates all other places**. This is where most real user-facing bugs live.

| Test | Action | Expected propagation |
|------|--------|---------------------|
| 14.1 | Edit line in group | Group detail API returns updated OCR text |
| 14.2 | Accept line in Review | Gallery `needs_review` count decreases |
| 14.3 | Switch board | Gallery + Queue + Review all show new board's data |
| 14.4 | Batch accept agreeing | Gallery filter auto-switches to `needs_review` |
| 14.5 | Batch accept selected | Gallery filter auto-switches to `needs_review` |
| 14.6 | Switch subdir (candidate↔final) | Gallery shows different data set |
| 14.7 | Rotate image | Gallery thumbnail URLs change (cache bust) |
| 14.8 | Accept instance in comparison | Gallery doesn't crash, reflects status |
| 14.9 | Delete line | Queue length decreases |
| 14.10 | Add line | Queue length increases |
| 14.11 | Position filter | Gallery scoped to selected position |

## Rules for Writing Tests

### Basics
1. **No skipping errors.** Every `assert` must pass. If an API returns `success: false`, the test fails.
2. **No "pre-existing bug" exemptions.** If a test exposes a bug, fix the bug, don't skip the test.
3. **Auto-dismiss dialogs.** Use `page.on('dialog', d => d.accept())` so `confirm()` calls don't block headless Chrome.
4. **Restore state.** Rotate CW then CCW. Delete what you add. Don't leave test artifacts.
5. **Use fresh boards.** For rotate tests, pick a board with no reviewed/edited lines to avoid the rotation block.

### Cross-Component Rules
6. **Every mutation must verify propagation.** If you add a new endpoint that modifies data, add a test that checks every other view that displays that data.
7. **Test filter/state resets.** After batch operations, verify that filters auto-switch (e.g., to `needs_review`). After board switch, verify gallery filter resets to `all`.
8. **Test the gallery cache.** Any operation that changes data must call `invalidateGalleryCache()`. Test that gallery shows fresh data, not stale cache.
9. **Test with real group structure.** Don't just test queue lines — find lines inside gallery groups and edit those, because gallery uses `majority_text` from component groups.
10. **Be aware of state leaking between tests.** If test N changes a filter to `needs_review`, test N+1's board switch must not see an empty gallery because the filter persisted.

### Data Integrity
11. **Test the full roundtrip.** After edit/add, re-fetch and verify the data actually changed. After delete, verify it's gone.
12. **Cover null/None edge cases.** JSON fields like `group_id`, `imageWidth`, `imageHeight` can be `null`. Tests must exercise code paths where these are absent.

## When to Run

- After **any** change to `ocr_review_app.py` or `templates/ocr_review.html`
- After refactoring helpers, loader, or event store
- Before committing

## Adding New Tests

Add new `test('name', async () => { ... })` blocks to `tests/test_e2e.js`. Follow the existing pattern:

```javascript
await test('X.Y Description of user action', async () => {
    const r = await api(page, 'POST', '/api/endpoint', { param: value });
    assert(r.success, `Failed: ${r.error}`);
    assert(r.some_field > 0, `Unexpected: ${r.some_field}`);
});
```

### Checklist for new features

When adding a new feature, ask:

- [ ] Does this mutation (edit/add/delete/accept/rotate) invalidate the gallery cache?
- [ ] Does this mutation rebuild component groups if it changes annotation content?
- [ ] Does this mutation mark metrics as dirty?
- [ ] After this operation, does switching to Gallery/Review/Dashboard show correct data?
- [ ] Does this operation work when `group_id` is null in JSON?
- [ ] Does this operation work when `imageWidth`/`imageHeight` is null?
- [ ] Is there a cross-component test verifying propagation?
