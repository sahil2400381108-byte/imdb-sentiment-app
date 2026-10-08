"""End-to-end browser checks against a running local application."""
import json
from pathlib import Path

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8000"
VIEWS = ("analyze", "evaluation", "method")


def assert_view(page, view):
    """A route must expose only its content and identify its navigation link."""
    expect(page.locator(f"#{view}")).to_be_visible()
    expect(page.locator(f'[data-nav="{view}"]').first).to_have_attribute("aria-current", "page")
    for other in VIEWS:
        if other != view:
            expect(page.locator(f"#{other}")).to_be_hidden()


def screenshot(page, name):
    """Wait for visible artwork and fonts so captures show the finished view."""
    page.evaluate("document.fonts.ready")
    page.wait_for_function("""() => [...document.images]
        .filter(image => image.getClientRects().length)
        .every(image => image.complete && image.naturalWidth > 0)""")
    page.evaluate("window.scrollTo({top: 0, behavior: 'instant'})")
    page.screenshot(path=str(ROOT / f"results/{name}.png"), full_page=True)


def check_download(page, filename):
    link = page.locator(f'a[href="/api/download/{filename}"]')
    details = link.locator("xpath=ancestor::details")
    expand = details.count() and details.get_attribute("open") is None
    if expand:
        details.locator("summary").click()
    with page.expect_download() as download:
        link.click()
    artifact = download.value
    assert artifact.suggested_filename == filename
    assert artifact.failure() is None
    assert Path(artifact.path()).read_bytes() == (ROOT / "results" / filename).read_bytes()
    if expand:
        details.locator("summary").click()


def main():
    metrics = json.loads((ROOT / "results/metrics.json").read_text(encoding="utf-8"))
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width":1440,"height":1050}, device_scale_factor=1, reduced_motion="reduce")
        errors = []
        broken_assets = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("response", lambda response: broken_assets.append(response.url)
                if "/static/" in response.url and response.status >= 400 else None)
        page.goto(BASE)
        expect(page.locator("#model-status")).to_have_text("Model ready")
        assert_view(page, "analyze")
        expect(page.locator("#prediction h3")).to_have_text("Ready when you are")
        assert page.evaluate("""() => {
            const rgb = getComputedStyle(document.body).backgroundColor.match(/[\\d.]+/g).slice(0, 3);
            const linear = rgb.map(value => {
                const color = Number(value) / 255;
                return color <= .04045 ? color / 12.92 : ((color + .055) / 1.055) ** 2.4;
            });
            return .2126 * linear[0] + .7152 * linear[1] + .0722 * linear[2] < .1;
        }"""), "The cinema theme must retain a dark page background"
        screenshot(page, "screenshot-desktop")
        page.locator("#write-review-button").click()
        expect(page.locator("#review")).to_be_focused()
        assert_view(page, "analyze")
        page.get_by_role("button", name="Analyze sentiment").click()
        expect(page.locator("#input-error")).to_contain_text("Write or paste")
        expect(page.locator("#review")).to_have_attribute("aria-invalid", "true")
        page.locator('[data-example="positive"]').click()
        page.get_by_role("button", name="Analyze sentiment").click()
        expect(page.locator("#prediction h3")).to_have_text("Positive Sentiment")
        screenshot(page, "screenshot-prediction")
        page.locator('[data-example="negative"]').click()
        expect(page.locator("#prediction h3")).to_have_text("Ready when you are")
        page.locator("#review").press("Control+Enter")
        expect(page.locator("#prediction h3")).to_have_text("Negative Sentiment")
        page.locator('[data-example="mixed"]').click()
        expect(page.locator("#review")).to_be_focused()
        assert len(page.locator("#review").input_value()) > 20
        expect(page.locator("#prediction h3")).to_have_text("Ready when you are")
        page.get_by_role("button", name="Clear", exact=True).click()
        expect(page.locator("#review")).to_have_value("")
        expect(page.locator("#prediction h3")).to_have_text("Ready when you are")
        expect(page.locator("#review")).to_be_focused()
        page.locator("#review").fill("!!!")
        page.get_by_role("button", name="Analyze sentiment").click()
        expect(page.locator("#input-error")).to_contain_text("containing words")
        page.locator("#review").fill("qzxwvvbnnqqrr")
        page.get_by_role("button", name="Analyze sentiment").click()
        expect(page.locator("#input-error")).to_contain_text("No familiar vocabulary")
        # Simulate temporary server failure, then confirm recovery.
        page.route("**/api/predict", lambda route: route.abort())
        page.locator('[data-example="positive"]').click()
        page.get_by_role("button", name="Analyze sentiment").click()
        expect(page.locator("#input-error")).to_contain_text("Cannot reach")
        page.unroute("**/api/predict")
        page.get_by_role("button", name="Analyze sentiment").click()
        expect(page.locator("#prediction h3")).to_have_text("Positive Sentiment")
        page.locator('[data-nav="evaluation"]').click()
        assert_view(page, "evaluation")
        for key, value in metrics["metrics"].items():
            expect(page.locator(f'#evaluation [data-metric="{key}"]')).to_have_text(f"{value*100:.2f}%")
        expect(page.locator("#report-body tr")).to_have_count(4)
        for cell, value in zip(("tn", "fp", "fn", "tp"), sum(metrics["confusion_matrix"], [])):
            expect(page.locator(f'#live-matrix [data-matrix-cell="{cell}"]')).to_have_text(str(value))
        boxes = {cell: page.locator(f'[data-matrix-cell="{cell}"]').locator("..").bounding_box()
                 for cell in ("tn", "fp", "fn", "tp")}
        assert boxes["tn"]["x"] == boxes["fn"]["x"] < boxes["fp"]["x"] == boxes["tp"]["x"]
        assert boxes["tn"]["y"] == boxes["fp"]["y"] < boxes["fn"]["y"] == boxes["tp"]["y"]
        page.locator(".matrix-export summary").click()
        expect(page.locator("#matrix-image")).to_be_visible()
        page.wait_for_function("document.querySelector('#matrix-image').naturalWidth > 0")
        page.locator(".matrix-export summary").click()
        screenshot(page, "screenshot-evaluation")
        for filename in ("metrics.json", "confusion_matrix.png", "classification_report.txt"):
            check_download(page, filename)
        page.locator('[data-nav="method"]').click()
        assert_view(page, "method")
        expect(page.locator("#dataset-total")).to_have_text(f'{metrics["dataset"]["final_records"]:,}')
        for sentiment in ("positive", "negative"):
            expect(page.locator(f"#{sentiment}-count")).to_have_text(
                f'{metrics["dataset"]["class_distribution"][sentiment]:,}')
        check_download(page, "dataset_audit.json")
        screenshot(page, "screenshot-methodology")
        for width in [390, 360, 768]:
            page.set_viewport_size({"width":width,"height":844})
            for view in VIEWS:
                page.goto(f"{BASE}/#{view}")
                expect(page.locator("#model-status")).to_have_text("Model ready")
                assert_view(page, view)
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), f"Overflow: {width}/{view}"
                if width == 390:
                    # Capture a resting view, without keyboard focus restored by hash navigation.
                    page.evaluate("document.activeElement.blur()")
                    screenshot(page, f"screenshot-mobile-{view}")
        page.set_viewport_size({"width":390,"height":844})
        page.goto(f"{BASE}/#unknown-view")
        assert_view(page, "analyze")
        # Navigation controls remain usable on a phone, as do movie-review inputs.
        for view in ("evaluation", "method", "analyze"):
            page.locator(f'[data-nav="{view}"]').click()
            assert_view(page, view)
        page.locator("#write-review-button").click()
        expect(page.locator("#review")).to_be_focused()
        page.locator('[data-example="negative"]').click()
        page.get_by_role("button", name="Analyze sentiment").click()
        expect(page.locator("#prediction h3")).to_have_text("Negative Sentiment")
        # A phone user should see the verdict without manually scrolling past the editor.
        page.wait_for_function("""() => {
            const verdict = document.querySelector('.result-card').getBoundingClientRect();
            const header = document.querySelector('.site-header').getBoundingClientRect();
            return verdict.top >= header.bottom && verdict.bottom <= innerHeight;
        }""")
        page.screenshot(path=str(ROOT / "results/screenshot-mobile-verdict.png"))
        # Initialization failure must be visible and retryable.
        page.route("**/api/experiment", lambda route: route.fulfill(status=503, body="Unavailable"))
        page.reload()
        expect(page.locator("#load-error")).to_be_visible()
        expect(page.locator("#analyze-button")).to_be_disabled()
        page.unroute("**/api/experiment")
        page.get_by_role("button", name="Try again").click()
        expect(page.locator("#model-status")).to_have_text("Model ready")
        assert not errors, errors
        assert not broken_assets, broken_assets
        # Motion is subtle in normal mode and fully disabled when requested.
        motion_page = browser.new_page(reduced_motion="no-preference")
        motion_page.goto(BASE)
        assert motion_page.locator(".hero-art").evaluate("element => getComputedStyle(element).animationName") == "cinema-settle"
        motion_page.emulate_media(reduced_motion="reduce")
        assert motion_page.locator(".hero-art").evaluate("element => getComputedStyle(element).animationName") == "none"
        assert motion_page.locator(".primary").first.evaluate("element => getComputedStyle(element).transitionDuration") == "0s"
        motion_page.close()
        browser.close()
    result = {"browser":"Chromium", "passed":True, "javascript_errors":errors, "broken_static_assets":broken_assets, "viewports":[1440,768,390,360], "checks":["dark cinema theme", "hero review CTA focus", "three sample review cards", "positive and negative predictions", "blank/symbol/unknown vocabulary validation", "clear and stale-result reset", "keyboard submission", "network failure and recovery", "experiment retry", "metrics and confusion matrix match saved JSON", "visible imagery renders", "classification report", "all four downloads match saved artifacts", "active navigation and direct links", "unknown route fallback", "mobile navigation, overflow and prediction", "mobile verdict automatically visible", "normal animation and reduced-motion support"]}
    (ROOT / "results/browser_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
