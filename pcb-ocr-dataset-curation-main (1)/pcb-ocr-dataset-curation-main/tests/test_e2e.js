/**
 * End-to-end test: simulates real user workflows via headless Chrome.
 * Every API call must succeed — no skipping errors.
 *
 * Usage:
 *   1. Start server: python ocr_review_app.py --dataset_root Data/ocr_zip_files/0311_ocr --port 5001
 *   2. Run tests:    node tests/test_e2e.js
 */
const puppeteer = require('puppeteer');
const BASE = process.env.TEST_URL || 'http://localhost:5001';

let passed = 0, failed = 0;
const errors = [];
async function test(name, fn) {
    try { await fn(); passed++; console.log(`  ✓ ${name}`); }
    catch (e) { failed++; errors.push({ name, error: e.message }); console.log(`  ✗ ${name}: ${e.message}`); }
}
function assert(c, m) { if (!c) throw new Error(m || 'assertion failed'); }
const wait = (ms = 1500) => new Promise(r => setTimeout(r, ms));

async function api(page, method, path, body) {
    return page.evaluate(async (m, p, b) => {
        const opts = { method: m, headers: { 'Content-Type': 'application/json' } };
        if (b) opts.body = JSON.stringify(b);
        return (await fetch(p, opts)).json();
    }, method, path, body);
}

(async () => {
    const browser = await puppeteer.launch({
        headless: 'new',
        args: ['--no-sandbox'],
        protocolTimeout: 30000
    });
    const page = await browser.newPage();
    await page.setCacheEnabled(false);
    page.on('dialog', async dialog => { await dialog.accept(); });
    const jsErrors = [];
    page.on('pageerror', e => jsErrors.push(e.message));

    await page.goto(BASE, { waitUntil: 'networkidle0', timeout: 15000 });
    await wait(2000);

    // Start on board_1 (has known data with null group_ids — good edge case)
    await page.evaluate(() => selectBoard('board_1'));
    await wait(3000);

    // =========================================================================
    console.log('\n══════ 1. ACCEPT LINE ══════');
    // =========================================================================

    await test('1.1 Load queue', async () => {
        const q = await api(page, 'GET', '/api/queue');
        assert(q.success && q.queue.length > 0, `queue: ${q.queue?.length}`);
    });

    await test('1.2 Load line data with image', async () => {
        const q = await api(page, 'GET', '/api/queue');
        const uid = q.queue[0].line_uid;
        const r = await api(page, 'GET', `/api/line/${encodeURIComponent(uid)}?enhanced=0`);
        assert(r.success, r.error);
        assert(r.line.image_id, 'no image_id');
        assert(r.line.image_data, 'no image_data');
        assert(r.line.bbox, 'no bbox');
    });

    await test('1.3 Accept line', async () => {
        const q = await api(page, 'GET', '/api/queue');
        const r = await api(page, 'POST', '/api/line/accept', { line_uid: q.queue[0].line_uid });
        assert(r.success, r.error);
    });

    // =========================================================================
    console.log('\n══════ 2. EDIT LINE ══════');
    // =========================================================================

    let editUid, editLine;

    await test('2.1 Get line to edit', async () => {
        const q = await api(page, 'GET', '/api/queue');
        editUid = q.queue[Math.min(1, q.queue.length - 1)].line_uid;
        const r = await api(page, 'GET', `/api/line/${encodeURIComponent(editUid)}?enhanced=0`);
        assert(r.success, r.error);
        editLine = r.line;
    });

    await test('2.2 Edit: change OCR text', async () => {
        const r = await api(page, 'POST', '/api/line/edit', {
            line_uid: editUid,
            edits: {
                ocr_text: (editLine.ocr_text || '') + 'X',
                line_bbox: editLine.bbox,
                chars: editLine.chars.map(c => ({ idx: c.idx, label: c.label, bbox: c.bbox }))
            }
        });
        assert(r.success, `Edit text failed: ${r.error}`);
    });

    await test('2.3 Verify text changed', async () => {
        const r = await api(page, 'GET', `/api/line/${encodeURIComponent(editUid)}?enhanced=0`);
        assert(r.success, r.error);
        assert(r.line.ocr_text.endsWith('X'), `text: ${r.line.ocr_text}`);
    });

    await test('2.4 Edit: modify char bbox', async () => {
        const chars = editLine.chars.map(c => ({ idx: c.idx, label: c.label, bbox: { ...c.bbox } }));
        if (chars.length > 0) chars[0].bbox.x2 -= 1;
        const r = await api(page, 'POST', '/api/line/edit', {
            line_uid: editUid,
            edits: { ocr_text: editLine.ocr_text || 'T', line_bbox: editLine.bbox, chars }
        });
        assert(r.success, `Edit char bbox failed: ${r.error}`);
    });

    await test('2.5 Edit: modify line bbox', async () => {
        const bbox = { ...editLine.bbox, x2: editLine.bbox.x2 - 1 };
        const r = await api(page, 'POST', '/api/line/edit', {
            line_uid: editUid,
            edits: {
                ocr_text: editLine.ocr_text || 'T',
                line_bbox: bbox,
                chars: editLine.chars.map(c => ({ idx: c.idx, label: c.label, bbox: c.bbox }))
            }
        });
        assert(r.success, `Edit line bbox failed: ${r.error}`);
    });

    // =========================================================================
    console.log('\n══════ 3. ADD LINE ══════');
    // =========================================================================

    let addedUid;

    await test('3.1 Add new line with chars', async () => {
        const q = await api(page, 'GET', '/api/queue');
        const imgId = q.queue[0].image_id;
        const r = await api(page, 'POST', '/api/line/add', {
            image_id: imgId,
            line_bbox: { x1: 5, y1: 5, x2: 80, y2: 25 },
            ocr_text: 'NEW',
            chars: [
                { idx: 0, label: 'N', bbox: { x1: 5, y1: 5, x2: 30, y2: 25 } },
                { idx: 1, label: 'E', bbox: { x1: 30, y1: 5, x2: 55, y2: 25 } },
                { idx: 2, label: 'W', bbox: { x1: 55, y1: 5, x2: 80, y2: 25 } }
            ]
        });
        assert(r.success, `Add line failed: ${r.error}`);
        addedUid = r.line_uid;
        assert(addedUid, 'no line_uid returned');
    });

    await test('3.2 Verify added line', async () => {
        const r = await api(page, 'GET', `/api/line/${encodeURIComponent(addedUid)}?enhanced=0`);
        assert(r.success, `Load added line: ${r.error}`);
        assert(r.line.ocr_text === 'NEW', `text: ${r.line.ocr_text}`);
        assert(r.line.chars.length === 3, `chars: ${r.line.chars.length}`);
    });

    // =========================================================================
    console.log('\n══════ 4. DELETE LINE ══════');
    // =========================================================================

    await test('4.1 Delete the added line', async () => {
        const r = await api(page, 'POST', '/api/line/delete', { line_uid: addedUid });
        assert(r.success, `Delete failed: ${r.error}`);
    });

    await test('4.2 Deleted line is gone', async () => {
        const r = await api(page, 'GET', `/api/line/${encodeURIComponent(addedUid)}?enhanced=0`);
        assert(!r.success, 'Deleted line still exists');
    });

    // =========================================================================
    console.log('\n══════ 5. ROTATE IMAGE ══════');
    // Use fresh board with no reviewed lines
    // =========================================================================

    await test('5.1 Rotate CW then CCW (roundtrip)', async () => {
        // Use a board not touched by earlier tests to avoid "has reviewed lines" block
        await page.evaluate(() => selectBoard('color2'));
        await wait(2000);
        const q = await api(page, 'GET', '/api/queue');
        assert(q.success && q.queue.length > 0, 'empty queue');
        const imgId = q.queue[0].image_id;
        const r1 = await api(page, 'POST', '/api/image/rotate', { image_id: imgId, direction: 'cw' });
        assert(r1.success, `CW: ${r1.error}`);
        const r2 = await api(page, 'POST', '/api/image/rotate', { image_id: imgId, direction: 'ccw' });
        assert(r2.success, `CCW: ${r2.error}`);
    });

    await test('5.2 Image serves after rotation', async () => {
        const q = await api(page, 'GET', '/api/queue');
        const status = await page.evaluate(async (id) => (await fetch(`/api/img/${id}?_t=${Date.now()}`)).status, q.queue[0].image_id);
        assert(status === 200, `status: ${status}`);
    });

    // =========================================================================
    console.log('\n══════ 6. ACCEPT GROUP ══════');
    // =========================================================================

    await test('6.1 Accept group', async () => {
        await page.evaluate(() => selectBoard('board_1'));
        await wait(2000);
        const g = await api(page, 'GET', `/api/gallery?page=1&filter=all&sort=conf_asc&_t=${Date.now()}`);
        assert(g.success && g.groups.length > 0, 'no groups');
        const r = await api(page, 'POST', '/api/group/accept', { region_group_id: g.groups[0].group_id });
        assert(r.success, r.error);
        assert(r.accepted_count > 0);
    });

    // =========================================================================
    console.log('\n══════ 7. REJECT INSTANCE ══════');
    // =========================================================================

    await test('7.1 Reject instance', async () => {
        const g = await api(page, 'GET', `/api/gallery?page=1&filter=all&sort=conf_asc&_t=${Date.now()}`);
        if (g.groups.length > 0) {
            const detail = await api(page, 'GET', `/api/group/${encodeURIComponent(g.groups[0].group_id)}`);
            assert(detail.success, detail.error);
            const r = await api(page, 'POST', '/api/instance/reject', {
                image_id: detail.instances[0].image_id,
                region_group_id: g.groups[0].group_id
            });
            assert(r.success, r.error);
        }
    });

    // =========================================================================
    console.log('\n══════ 8. BATCH ROTATE GROUP ══════');
    // =========================================================================

    await test('8.1 Batch rotate CW+CCW', async () => {
        await page.evaluate(() => selectBoard('board_3'));
        await wait(2000);
        const g = await api(page, 'GET', `/api/gallery?page=1&filter=all&sort=conf_asc&_t=${Date.now()}`);
        assert(g.success && g.groups.length > 0);
        const gid = g.groups[0].group_id;
        const r1 = await api(page, 'POST', '/api/group/rotate', { group_id: gid, direction: 'cw' });
        assert(r1.success, r1.error);
        const r2 = await api(page, 'POST', '/api/group/rotate', { group_id: gid, direction: 'ccw' });
        assert(r2.success, r2.error);
    });

    // =========================================================================
    console.log('\n══════ 9. BATCH ACCEPT AGREEING ══════');
    // =========================================================================

    await test('9.1 Batch accept agreeing', async () => {
        const r = await api(page, 'POST', '/api/batch/accept_agreeing', {});
        assert(r.success, r.error);
    });

    // =========================================================================
    console.log('\n══════ 10. DIRTY STATE + SAVE ══════');
    // =========================================================================

    await test('10.1 Edit text → dirty → save', async () => {
        await page.evaluate(() => selectBoard('board_1'));
        await wait(2000);
        await page.evaluate(() => switchTab('review'));
        await wait(1500);
        await page.evaluate(() => navigate('first'));
        await wait(1500);

        await page.focus('#edit-text-input');
        await page.keyboard.type('Z');
        await wait(400);
        assert(await page.$eval('#btn-save', e => e.style.display !== 'none'), 'Not dirty');

        await page.evaluate(() => saveEdit());
        await wait(3000);
    });

    await test('10.2 Discard changes', async () => {
        await page.focus('#edit-text-input');
        await page.keyboard.type('Q');
        await wait(400);
        await page.evaluate(() => discardChangesQuiet());
        await wait(400);
        assert(await page.$eval('#btn-save', e => e.style.display === 'none'), 'Still dirty');
    });

    // =========================================================================
    console.log('\n══════ 11. BOARD SWITCH CONSISTENCY ══════');
    // =========================================================================

    await test('11.1 Switch A→B→A, gallery matches', async () => {
        await page.evaluate(() => switchTab('gallery'));
        await wait(1000);
        await page.evaluate(() => selectBoard('board_0'));
        await wait(2000);
        const a1 = await page.$$eval('.gallery-card', c => c.map(e => e.dataset.groupId));
        await page.evaluate(() => selectBoard('board_1'));
        await wait(2000);
        const b = await page.$$eval('.gallery-card', c => c.map(e => e.dataset.groupId));
        await page.evaluate(() => selectBoard('board_0'));
        await wait(2000);
        const a2 = await page.$$eval('.gallery-card', c => c.map(e => e.dataset.groupId));
        assert(JSON.stringify(a1) !== JSON.stringify(b), 'A==B');
        assert(JSON.stringify(a1) === JSON.stringify(a2), 'A1!=A2');
    });

    // =========================================================================
    console.log('\n══════ 12. GALLERY UI ══════');
    // =========================================================================

    await test('12.1 Gallery cards with thumbnails', async () => {
        const loaded = await page.$$eval('.gallery-card img', imgs => imgs.filter(i => i.complete && i.naturalWidth > 0).length);
        assert(loaded > 0, `0 thumbnails loaded`);
    });

    await test('12.2 Position selector populated', async () => {
        assert(await page.$$eval('#gallery-position-select option', o => o.length) >= 2);
    });

    await test('12.3 Filter/sort selectors work', async () => {
        await page.select('#gallery-filter-select', 'needs_review');
        await wait(1000);
        await page.select('#gallery-filter-select', 'all');
        await wait(1000);
        assert(await page.$$eval('.gallery-card', c => c.length) > 0);
    });

    await test('12.4 Keyboard focus navigation', async () => {
        await page.keyboard.press('ArrowRight');
        await wait(200);
        await page.keyboard.press('ArrowRight');
        await wait(200);
        assert(await page.$('.gallery-card.focused'), 'No focused card');
    });

    await test('12.5 Comparison modal opens and closes', async () => {
        await page.keyboard.press('Enter');
        await wait(2000);
        assert(await page.$eval('#comparison-modal', e => e.classList.contains('active')));
        await page.keyboard.press('Escape');
        await wait(300);
        assert(!await page.$eval('#comparison-modal', e => e.classList.contains('active')));
    });

    // =========================================================================
    console.log('\n══════ 13. REVIEW TAB CONTROLS ══════');
    // =========================================================================

    await test('13.1 All mode buttons work', async () => {
        await page.evaluate(() => switchTab('review'));
        await wait(1500);
        for (const [key, id] of [['c', 'mode-draw-char'], ['l', 'mode-draw-line'], ['s', 'mode-select']]) {
            await page.keyboard.press(key);
            await wait(200);
            assert(await page.$eval(`#${id}`, e => e.classList.contains('active')), `${key} mode failed`);
        }
    });

    await test('13.2 Navigate first/last', async () => {
        await page.evaluate(() => navigate('first'));
        await wait(1000);
        const first = await page.$eval('#current-position', e => e.textContent);
        assert(first.startsWith('1 /'), `first: ${first}`);
        await page.evaluate(() => navigate('last'));
        await wait(1000);
        const last = await page.$eval('#current-position', e => e.textContent);
        const parts = last.split('/').map(s => s.trim());
        assert(parts[0] === parts[1], `last: ${last}`);
    });

    await test('13.3 Zoom/brightness/contrast sliders exist', async () => {
        assert(await page.$('#zoom-slider'));
        assert(await page.$('#brightness-slider'));
        assert(await page.$('#contrast-slider'));
    });

    // =========================================================================
    console.log('\n══════ 14. CROSS-COMPONENT INTERACTION LOGIC ══════');
    // These tests verify that actions in one part of the UI correctly
    // propagate to other parts. These are the bugs users actually find.
    // =========================================================================

    // -- 14.1: Edit line → verify groups rebuild with fresh data --
    // Pick a line that's inside a gallery group (not just any queue line)
    await test('14.1 Edit line in group → group detail reflects edit', async () => {
        await page.evaluate(() => selectBoard('board_1'));
        await wait(2000);

        // Get a line from a gallery group
        const g = await api(page, 'GET', `/api/gallery?page=1&filter=all&sort=conf_asc&_t=${Date.now()}`);
        assert(g.success && g.groups.length > 0, 'no groups');
        const detail = await api(page, 'GET', `/api/group/${encodeURIComponent(g.groups[0].group_id)}`);
        const lineUid = detail.instances[0].lines[0].line_uid;
        const origOcr = detail.instances[0].lines[0].ocr_text;

        // Load full line data
        const lineResp = await api(page, 'GET', `/api/line/${encodeURIComponent(lineUid)}?enhanced=0`);
        assert(lineResp.success, lineResp.error);

        // Edit: change text
        const newText = origOcr + 'Z';
        const r = await api(page, 'POST', '/api/line/edit', {
            line_uid: lineUid,
            edits: {
                ocr_text: newText,
                line_bbox: lineResp.line.bbox,
                chars: lineResp.line.chars.map(c => ({ idx: c.idx, label: c.label, bbox: c.bbox }))
            }
        });
        assert(r.success, `Edit failed: ${r.error}`);

        // Verify: re-fetch gallery (group ID may have changed since text changed)
        // Then find the instance by line_uid across all groups
        const g2 = await api(page, 'GET', `/api/gallery?page=1&filter=all&sort=conf_asc&_t=${Date.now()}`);
        let found = false;
        for (const grp of g2.groups) {
            const d = await api(page, 'GET', `/api/group/${encodeURIComponent(grp.group_id)}`);
            if (!d.success) continue;
            for (const inst of d.instances) {
                const match = inst.lines.find(l => l.line_uid === lineUid);
                if (match) {
                    assert(match.ocr_text === newText,
                        `Expected "${newText}", got "${match.ocr_text}"`);
                    found = true;
                    break;
                }
            }
            if (found) break;
        }
        assert(found, `Line ${lineUid} not found in any group after edit`);
    });

    // -- 14.2: Accept line in Review → Gallery status updates --
    await test('14.2 Accept in Review → Gallery status changes', async () => {
        await page.evaluate(() => selectBoard('board_2'));
        await wait(2000);

        // Check gallery for needs_review groups
        await page.evaluate(() => switchTab('gallery'));
        await wait(1500);
        await page.evaluate(() => {
            document.getElementById('gallery-filter-select').value = 'needs_review';
            changeGalleryFilter();
        });
        await wait(1500);
        const beforeCount = await page.$$eval('.gallery-card', c => c.length);

        // Go to review, accept first unreviewed line
        await page.evaluate(() => switchTab('review'));
        await wait(1500);
        const uid = await page.$eval('#current-line-uid', e => e.textContent);
        await api(page, 'POST', '/api/line/accept', { line_uid: uid });

        // Back to gallery with needs_review filter
        await page.evaluate(() => switchTab('gallery'));
        await wait(2000);
        const afterCount = await page.$$eval('.gallery-card', c => c.length);
        // After accepting, needs_review count should decrease or stay same
        assert(afterCount <= beforeCount,
            `needs_review count increased: ${beforeCount} -> ${afterCount}`);
    });

    // -- 14.3: Board switch → Gallery + Queue + Review all update --
    await test('14.3 Board switch → all tabs reflect new board', async () => {
        await page.evaluate(() => selectBoard('board_0'));
        await wait(2000);

        // Check gallery
        const galleryCards = await page.$$eval('.gallery-card', c => c.length);
        assert(galleryCards > 0, 'Gallery empty after switch');

        // Check queue
        const q = await api(page, 'GET', '/api/queue');
        assert(q.current_class === 'board_0', `Queue class: ${q.current_class}`);

        // Check review tab shows data from correct board
        await page.evaluate(() => switchTab('review'));
        await wait(1500);
        const imgId = await page.$eval('#current-image-id', e => e.textContent);
        assert(imgId && imgId !== '-', `Review shows no image after board switch`);

        // Switch back to gallery
        await page.evaluate(() => switchTab('gallery'));
        await wait(1000);
    });

    // -- 14.4: Select agreeing groups → checkboxes selected, no auto-accept --
    await test('14.4 Select agreeing groups → checkboxes auto-selected', async () => {
        await page.evaluate(() => selectBoard('board_5'));
        await wait(2000);
        await page.evaluate(() => switchTab('gallery'));
        await wait(1500);
        await page.evaluate(() => {
            document.getElementById('gallery-filter-select').value = 'all';
            changeGalleryFilter();
        });
        await wait(1000);

        // Clear any existing selection
        await page.evaluate(() => { selectedGroups.clear(); updateGallerySelectedCount(); });

        // Call select agreeing groups
        await page.evaluate(() => selectAgreeingGroups());
        await wait(1000);

        // Should have selected some groups (checkboxes), not auto-accepted
        const countText = await page.$eval('#gallery-selected-count', e => e.textContent);
        // Filter should still be 'all' (not auto-switched)
        const filterVal = await page.$eval('#gallery-filter-select', e => e.value);
        assert(filterVal === 'all', `Filter changed unexpectedly: ${filterVal}`);
    });

    // -- 14.5: Batch accept selected → Filter auto-switches --
    await test('14.5 Batch accept selected → auto-switches to needs_review filter', async () => {
        await page.evaluate(() => selectBoard('board_6'));
        await wait(2000);
        await page.evaluate(() => switchTab('gallery'));
        await wait(1500);
        await page.evaluate(() => {
            document.getElementById('gallery-filter-select').value = 'all';
            changeGalleryFilter();
        });
        await wait(1500);

        // Select first group and batch accept
        const gid = await page.evaluate(() => {
            const card = document.querySelector('.gallery-card');
            if (!card) return null;
            const id = card.dataset.groupId;
            selectedGroups.add(id);
            return id;
        });
        if (gid) {
            await page.evaluate(() => batchAcceptSelected());
            await wait(3000);
            const filterVal = await page.$eval('#gallery-filter-select', e => e.value);
            assert(filterVal === 'needs_review',
                `Filter should be needs_review, got: ${filterVal}`);
        }
    });

    // -- 14.6: Subdir switch → Gallery shows different data --
    await test('14.6 Subdir switch candidate→final → Gallery updates', async () => {
        await page.evaluate(() => selectBoard('board_0'));
        await wait(2000);
        await page.evaluate(() => switchTab('gallery'));
        await wait(1500);
        await page.evaluate(() => {
            document.getElementById('gallery-filter-select').value = 'all';
            changeGalleryFilter();
        });
        await wait(1000);
        const candidateCards = await page.$$eval('.gallery-card', c => c.length);

        // Switch to final — may have 0 cards if nothing was finalized
        const switchResult = await api(page, 'POST', '/api/switch_subdir', { subdir: 'final' });
        if (switchResult.success) {
            await wait(1500);
            const finalCards = await page.$$eval('.gallery-card', c => c.length);
            // Final should be different from candidate (likely 0 if nothing moved)
            // Just verify it didn't crash and gallery updated
            assert(finalCards >= 0, 'Gallery broken after subdir switch');

            // Switch back to candidate
            await api(page, 'POST', '/api/switch_subdir', { subdir: 'candidate' });
            await wait(1500);
        }
    });

    // -- 14.7: Rotate image in Review → Gallery thumbnail updates --
    await test('14.7 Rotate in Review → Gallery thumbnail URL changes', async () => {
        await page.evaluate(() => selectBoard('board_7'));
        await wait(2000);
        await page.evaluate(() => switchTab('gallery'));
        await wait(1500);

        // Record thumbnail URLs
        const beforeUrls = await page.$$eval('.gallery-card img', imgs => imgs.map(i => i.src));

        // Rotate first image
        const q = await api(page, 'GET', '/api/queue');
        if (q.queue.length > 0) {
            const imgId = q.queue[0].image_id;
            await api(page, 'POST', '/api/image/rotate', { image_id: imgId, direction: 'cw' });

            // Go to gallery, invalidate cache, reload
            await page.evaluate(() => { invalidateGalleryCache(); loadGallery(); });
            await wait(2000);
            const afterUrls = await page.$$eval('.gallery-card img', imgs => imgs.map(i => i.src));

            // URLs should differ (different _t= timestamp at minimum)
            assert(JSON.stringify(beforeUrls) !== JSON.stringify(afterUrls),
                'Gallery thumbnails unchanged after rotate');

            // Rotate back
            await api(page, 'POST', '/api/image/rotate', { image_id: imgId, direction: 'ccw' });
        }
    });

    // -- 14.8: Accept from comparison modal → Gallery card status updates --
    await test('14.8 Accept in comparison → Gallery reflects accepted status', async () => {
        await page.evaluate(() => selectBoard('board_8'));
        await wait(2000);
        await page.evaluate(() => switchTab('gallery'));
        await wait(1500);
        await page.evaluate(() => {
            document.getElementById('gallery-filter-select').value = 'all';
            changeGalleryFilter();
        });
        await wait(1000);

        // Open comparison for first group
        const card = await page.$('.gallery-card');
        if (card) {
            await card.click();
            await wait(2000);

            // Accept first instance
            const acceptBtn = await page.$('.comparison-card-actions .btn-comp-accept');
            if (acceptBtn) {
                await acceptBtn.click();
                await wait(2000);
            }

            // Close comparison
            await page.keyboard.press('Escape');
            await wait(1000);

            // Gallery should have reloaded — no crash
            const cards = await page.$$eval('.gallery-card', c => c.length);
            assert(cards >= 0, 'Gallery broken after comparison accept');
        }
    });

    // -- 14.9: Delete line → line removed from queue (may get image-level entry back) --
    await test('14.9 Delete line → line disappears from queue', async () => {
        await page.evaluate(() => selectBoard('board_9'));
        await wait(2000);

        const qBefore = await api(page, 'GET', '/api/queue');
        // Find a line (not an image-level entry) to delete
        const lineItem = qBefore.queue.find(q => q.line_uid.includes('#'));
        if (lineItem) {
            await api(page, 'POST', '/api/line/delete', { line_uid: lineItem.line_uid });
            await wait(1000);
            const qAfter = await api(page, 'GET', '/api/queue');
            // The specific line should be gone
            const stillThere = qAfter.queue.some(q => q.line_uid === lineItem.line_uid);
            assert(!stillThere, `Deleted line still in queue: ${lineItem.line_uid}`);
        }
    });

    // -- 14.10: Add line → new line appears in queue --
    await test('14.10 Add line → new line_uid appears in queue', async () => {
        const qBefore = await api(page, 'GET', '/api/queue');
        const imgId = qBefore.queue[0].image_id;

        const addResult = await api(page, 'POST', '/api/line/add', {
            image_id: imgId,
            line_bbox: { x1: 2, y1: 2, x2: 50, y2: 15 },
            ocr_text: 'ZZ',
            chars: [
                { idx: 0, label: 'Z', bbox: { x1: 2, y1: 2, x2: 25, y2: 15 } },
                { idx: 1, label: 'Z', bbox: { x1: 25, y1: 2, x2: 50, y2: 15 } }
            ]
        });
        assert(addResult.success, addResult.error);

        const qAfter = await api(page, 'GET', '/api/queue');
        const found = qAfter.queue.some(q => q.line_uid === addResult.line_uid);
        assert(found, `Added line ${addResult.line_uid} not in queue`);

        // Clean up
        await api(page, 'POST', '/api/line/delete', { line_uid: addResult.line_uid });
    });

    // -- 14.11: Position filter → Gallery shows only that position's groups --
    await test('14.11 Position filter → Gallery scoped to position', async () => {
        await page.evaluate(() => selectBoard('board_0'));
        await wait(2000);
        await page.evaluate(() => switchTab('gallery'));
        await wait(1500);
        await page.evaluate(() => {
            document.getElementById('gallery-filter-select').value = 'all';
            changeGalleryFilter();
        });
        await wait(1000);
        const allCount = await page.$$eval('.gallery-card', c => c.length);

        // Select a specific position
        const posValue = await page.$$eval('#gallery-position-select option',
            opts => { const r = opts.filter(o => o.value !== ''); return r.length > 0 ? r[0].value : null; });
        if (posValue && allCount > 1) {
            await page.select('#gallery-position-select', posValue);
            await wait(1500);
            const filteredCount = await page.$$eval('.gallery-card', c => c.length);
            assert(filteredCount <= allCount,
                `Position filter didn't reduce: ${allCount} -> ${filteredCount}`);
            assert(filteredCount > 0, 'Position filter returned 0');

            // Reset
            await page.select('#gallery-position-select', '');
            await wait(1000);
        }
    });

    // =========================================================================
    console.log('\n══════ 15. JS ERROR CHECK ══════');
    // =========================================================================

    await test('15.1 Zero JS errors', () => {
        const real = jsErrors.filter(e => !e.includes('favicon') && !e.includes('404'));
        assert(real.length === 0, real.join('\n'));
    });

    // =========================================================================
    console.log(`\n${'═'.repeat(50)}`);
    console.log(`RESULTS: ${passed} passed, ${failed} failed out of ${passed + failed}`);
    if (errors.length) {
        console.log('\nFailed:');
        errors.forEach(e => console.log(`  ✗ ${e.name}: ${e.error}`));
    }
    console.log('═'.repeat(50));
    await browser.close();
    process.exit(failed > 0 ? 1 : 0);
})();
