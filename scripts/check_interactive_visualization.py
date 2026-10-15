"""Verify saved notebook hover plots in Chromium, including a sandboxed iframe.

Run with the optional ``browser`` dependency group and an installed Playwright
Chromium browser. This checks local HTML behavior, not a hosted Colab session.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import traceback
import uuid

import numpy as np


GRAPH = ".js-plotly-plot"
TIMEOUT_MS = 30_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--check-duplicate-fragments", action="store_true",
        help="Also check two copies with independent graph IDs in one document.",
    )
    return parser.parse_args()


def load_reference(representations: Path, provenance: Path) -> dict:
    with np.load(representations, allow_pickle=False) as saved:
        reference = {
            "waveforms": saved["waveforms"].astype(np.float64),
            "labels": saved["labels"].copy(),
            "projections": [saved[f"{kind}__tsne"].copy() for kind in ("random_init", "pretrained")],
        }
    reference["provenance"] = json.loads(provenance.read_text(encoding="utf-8"))
    waves, labels = reference["waveforms"], reference["labels"]
    assert waves.ndim == 3 and waves.shape[1] == 3 and waves.shape[2] > 0
    assert np.isfinite(waves).all() and labels.shape == (len(waves),)
    assert set(labels.tolist()) == {0, 1}
    assert len(reference["provenance"]["records"]) == len(waves)
    for projection in reference["projections"]:
        assert projection.shape == (len(waves), 2) and np.isfinite(projection).all()
    return reference


def verify_selection(frame, reference: dict, expected_id: int, selector=GRAPH) -> dict:
    frame.wait_for_function(
        "([selector, id]) => document.querySelector(selector).dataset.selectedId === String(id)",
        arg=[selector, expected_id],
        timeout=TIMEOUT_MS,
    )
    selected = frame.eval_on_selector(selector, """gd => ({
        id: Number(gd.dataset.selectedId), label: gd.dataset.selectedClass,
        trace: gd.dataset.selectedTrace,
        waves: gd.data.slice(6, 9).map(t => Array.from(t.y)),
        times: gd.data.slice(6, 9).map(t => Array.from(t.x)),
        range: gd.layout.yaxis3.range,
        highlights: gd.data.slice(4, 6).map(t => [t.x[0], t.y[0]])
    })""")
    waveform = reference["waveforms"][expected_id]
    expected = waveform + np.array([8, 0, -8])[:, None]
    np.testing.assert_allclose(selected["waves"], expected, rtol=0, atol=1e-12)
    sampling_rate = float(reference["provenance"]["sampling_rate_hz"])
    times = np.arange(waveform.shape[-1]) / sampling_rate
    np.testing.assert_allclose(selected["times"], np.tile(times, (3, 1)), rtol=0, atol=1e-12)
    assert selected["range"][0] <= expected.min() and selected["range"][1] >= expected.max()
    assert selected["label"] == ("Earthquake" if reference["labels"][expected_id] == 0 else "Noise")
    record = reference["provenance"]["records"][expected_id]
    assert selected["trace"] == (record.get("trace_name_original") or record.get("trace_name") or "recording")
    np.testing.assert_array_equal(selected["highlights"], [xy[expected_id] for xy in reference["projections"]])
    return {
        "waveform_id": expected_id,
        "matching_samples": int(expected.size),
        "matching_timestamps": int(times.size),
        "metadata_matches": True,
        "linked_highlights_match": True,
        "full_amplitude_visible": True,
    }


def pointer_candidates(frame, curve: int, previous_id: int, selector=GRAPH) -> list[dict]:
    # Prefer isolated points so a neighboring marker cannot capture the hover.
    return frame.eval_on_selector(selector, """(gd, [curve, previousId]) => {
        const trace = gd._fullData[curve];
        const xa = gd._fullLayout[trace.xaxis.replace('x', 'xaxis')];
        const ya = gd._fullLayout[trace.yaxis.replace('y', 'yaxis')];
        const rect = gd.getBoundingClientRect();
        const nearby = gd._fullData.slice(0, 4)
          .filter(t => t.xaxis === trace.xaxis && t.yaxis === trace.yaxis)
          .flatMap(t => Array.from(t.x, (x, i) => ({
              x: xa.l2p(x), y: ya.l2p(t.y[i]), id: Number(t.customdata[i])
          })));
        return Array.from(trace.x, (x, i) => {
            const px = xa.l2p(x), py = ya.l2p(trace.y[i]);
            const id = Number(trace.customdata[i]);
            const distances = nearby.filter(p => p.id !== id)
              .map(p => Math.hypot(px - p.x, py - p.y));
            return {point: i, id,
                x: rect.left + xa._offset + px,
                y: rect.top + ya._offset + py,
                clearance: distances.length ? Math.min(...distances) : 1000};
        }).filter(p => p.id !== previousId)
          .sort((a, b) => b.clearance - a.clearance).slice(0, 8);
    }""", [curve, previous_id])


def exercise(frame, page, reference: dict, mode: str, output_dir: Path, offset=(0, 0), selector=GRAPH) -> dict:
    frame.locator(selector).scroll_into_view_if_needed()
    frame.wait_for_function(
        "selector => document.querySelector(selector)?.dataset.ready === 'true'",
        arg=selector,
        timeout=TIMEOUT_MS,
    )
    checks = []
    for curve in range(4):
        previous_id = frame.eval_on_selector(selector, "gd => Number(gd.dataset.selectedId)")
        candidates = pointer_candidates(frame, curve, previous_id, selector)
        assert candidates, f"No distinct waveform is available in embedding trace {curve}."
        for target in candidates:
            page.mouse.move(1, 1)
            page.mouse.move(target["x"] + offset[0], target["y"] + offset[1])
            # A real pointer must select the expected global waveform index.
            try:
                frame.wait_for_function(
                    "([selector, id]) => document.querySelector(selector).dataset.selectedId === String(id)",
                    arg=[selector, target["id"]], timeout=2_000,
                )
                break
            except Exception:
                if target is candidates[-1]:
                    raise
        check = verify_selection(frame, reference, target["id"], selector)
        checks.append({"curve": curve, "local_point": target["point"], **check})

    # Also check the largest observed amplitude, regardless of its projection.
    extreme_id = int(np.argmax(np.max(np.abs(reference["waveforms"]), axis=(1, 2))))
    extreme_targets = frame.eval_on_selector(selector, """(gd, id) => {
        const targets = [], rect = gd.getBoundingClientRect();
        for (let curve = 0; curve < 4; curve++) {
            const point = Array.from(gd.data[curve].customdata).findIndex(v => Number(v) === id);
            if (point < 0) continue;
            const trace = gd._fullData[curve];
            const xa = gd._fullLayout[trace.xaxis.replace('x', 'xaxis')];
            const ya = gd._fullLayout[trace.yaxis.replace('y', 'yaxis')];
            targets.push({x: rect.left + xa._offset + xa.l2p(trace.x[point]),
                          y: rect.top + ya._offset + ya.l2p(trace.y[point])});
        }
        return targets;
    }""", extreme_id)
    assert extreme_targets, "Extreme-amplitude waveform is absent from the embedding."
    for target in extreme_targets:
        page.mouse.move(1, 1)
        page.mouse.move(target["x"] + offset[0], target["y"] + offset[1])
        try:
            frame.wait_for_function(
                "([selector, id]) => document.querySelector(selector).dataset.selectedId === String(id)",
                arg=[selector, extreme_id], timeout=2_000,
            )
            break
        except Exception:
            if target is extreme_targets[-1]:
                raise
    extreme_check = verify_selection(frame, reference, extreme_id, selector)
    page.screenshot(path=str(output_dir / f"{mode}.png"), full_page=True)
    return {"mode": mode, "actual_pointer_checks": checks, "extreme_amplitude_check": extreme_check}


def check_duplicate_fragments(page, html: str, graph_id: str, reference: dict, output_dir: Path) -> list[dict]:
    """Reproduce two notebook output fragments sharing one JavaScript context."""
    match = re.search(r"<body\b[^>]*>(.*)</body\s*>", html, flags=re.IGNORECASE | re.DOTALL)
    fragment = match.group(1) if match else html
    ids = [f"seislm-check-{uuid.uuid4().hex}" for _ in range(2)]
    selectors = [f'[id="{graph_id}"]' for graph_id in ids]
    page.set_content(
        '<!doctype html><html><head><meta charset="utf-8"></head><body style="margin:0">'
        + '<div style="height:24px"></div>'.join(fragment.replace(graph_id, new_id) for new_id in ids)
        + '</body></html>'
    )
    for selector in selectors:
        page.wait_for_function(
            "selector => document.querySelector(selector)?.dataset.ready === 'true'",
            arg=selector, timeout=TIMEOUT_MS,
        )
    assert page.locator('[data-seislm-interactive="true"]').count() == 2
    checks = []
    for index, selector in enumerate(selectors):
        other_selector = selectors[1 - index]
        previous_other = page.eval_on_selector(other_selector, "gd => gd.dataset.selectedId")
        checks.append(exercise(page, page, reference, f"duplicate-output-{index + 1}", output_dir, selector=selector))
        assert page.eval_on_selector(other_selector, "gd => gd.dataset.selectedId") == previous_other
        checks[-1]["other_output_unchanged"] = True
    return checks


def run(options: argparse.Namespace, report: dict) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as error:
        raise RuntimeError(
            "Install the browser dependency group and Chromium: "
            "uv sync --group browser; uv run --group browser playwright install chromium"
        ) from error

    reference = load_reference(options.representations, options.provenance)
    html = options.html.read_text(encoding="utf-8")
    errors, external_requests = report["browser_errors"], report["external_requests"]
    with sync_playwright() as tooling:
        browser = tooling.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1100}, offline=True)
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda request: external_requests.append(request.url)
                if request.url.startswith(("https://", "http://")) else None)
        page.goto(options.html.resolve().as_uri())
        graph_id = page.eval_on_selector(GRAPH, "gd => gd.id")
        report["checks"].append(exercise(page, page, reference, "standalone-offline", options.output_dir))
        page.set_content(
            '<html><body style="margin:0"><iframe name="notebook-output" '
            'sandbox="allow-scripts" style="border:0;width:1440px;height:1050px">'
            '</iframe></body></html>'
        )
        page.eval_on_selector("iframe", "(element, source) => { element.srcdoc = source; }", html)
        frame = page.frame(name="notebook-output")
        assert frame is not None
        box = page.locator("iframe").bounding_box()
        assert box is not None
        report["checks"].append(exercise(
            frame, page, reference, "sandbox-iframe-offline", options.output_dir,
            offset=(box["x"], box["y"]),
        ))
        if options.check_duplicate_fragments:
            report["checks"].extend(check_duplicate_fragments(page, html, graph_id, reference, options.output_dir))
        browser.close()
    assert not errors, f"Browser errors: {errors}"
    assert not external_requests, f"The saved visualization requested external resources: {external_requests}"
    report["passed"] = True


def main() -> int:
    options = parse_args()
    options.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {"html": options.html, "representations": options.representations, "provenance": options.provenance}
    report = {
        "passed": False, "hosted_colab_verified": False,
        "network": "offline", "iframe_sandbox": "allow-scripts, without allow-same-origin",
        "inputs": {name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for name, path in inputs.items()},
        "checks": [], "browser_errors": [], "external_requests": [],
    }
    try:
        run(options, report)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    destination = options.output_dir / "browser-verification.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "report": str(destination.resolve()),
                      "hosted_colab_verified": False}, indent=2))
    if not report["passed"]:
        print(report.get("traceback", "Verification failed."), file=sys.stderr)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
