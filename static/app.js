"use strict";
const $ = (selector) => document.querySelector(selector);
const review = $("#review");
const button = $("#analyze-button");
let ready = false;
let requestId = 0;
let controller = null;
const examples = {
  positive: "This movie was fantastic. The acting and story were excellent.",
  negative: "This movie was extremely boring and I regret watching it.",
  mixed: "The cast is talented, yet the story is dull and never gives them anything interesting to do."
};
const emptyResult = $("#prediction").innerHTML;
const descriptions = {accuracy: "All test predictions", precision: "Correct among predicted positives", recall: "Actual positives identified", f1: "Balance of precision and recall"};
const labels = {accuracy: "Accuracy", precision: "Precision", recall: "Recall", f1: "F1-score"};
const number = (value) => value.toLocaleString("en-US");
const percent = (value) => (value * 100).toFixed(2);
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

function focusReview() {
  review.focus({preventScroll: true});
  $("#review-workspace").scrollIntoView({behavior: reducedMotion.matches ? "instant" : "smooth", block: "start"});
}
$("#write-review-button").addEventListener("click", focusReview);

function navigate() {
  const valid = ["analyze", "evaluation", "method"];
  const target = valid.includes(location.hash.slice(1)) ? location.hash.slice(1) : "analyze";
  document.querySelectorAll(".view").forEach((view) => { view.hidden = view.id !== target; });
  document.querySelectorAll("[data-nav]").forEach((link) => {
    const active = link.dataset.nav === target;
    link.classList.toggle("active", active);
    if (active) link.setAttribute("aria-current", "page"); else link.removeAttribute("aria-current");
  });
  $("#breadcrumb").textContent = {analyze: "Review room", evaluation: "Model performance", method: "Behind the model"}[target];
}
window.addEventListener("hashchange", navigate);
navigate();

function metricCards(metrics) {
  return Object.keys(labels).map((key) => `<article class="metric"><div class="metric-label">${labels[key]}</div><div class="metric-value" data-metric="${key}">${percent(metrics[key])}<small>%</small></div><p>${descriptions[key]}</p></article>`).join("");
}

async function loadExperiment() {
  $("#load-error").hidden = true;
  try {
    const response = await fetch("/api/experiment", {signal: AbortSignal.timeout(15000)});
    if (!response.ok) throw new Error("Could not load the saved experiment. Check that the application server is running.");
    const data = await response.json();
    const {metrics, split, dataset, classification_report: report} = data;
    $("#preview-metrics").innerHTML = metricCards(metrics);
    $("#evaluation-metrics").innerHTML = metricCards(metrics);
    $("#holdout-summary").textContent = `${number(split.testing_records)} held-out reviews · Stratified 80/20 split · Positive-class precision, recall & F1`;
    $("#evaluation-summary").textContent = `${number(split.testing_records)} test reviews · ${number(split.testing_classes.negative)} negative / ${number(split.testing_classes.positive)} positive · Fixed seed 42 · No test-set tuning`;
    const [[tn, fp], [fn, tp]] = data.confusion_matrix;
    for (const [cell, count] of Object.entries({tn, fp, fn, tp})) {
      $(`[data-matrix-cell="${cell}"]`).textContent = number(count);
    }
    $("#matrix-image").alt = `Confusion matrix. Actual negative: ${tn} predicted negative, ${fp} predicted positive. Actual positive: ${fn} predicted negative, ${tp} predicted positive.`;
    $("#live-matrix").setAttribute("aria-label", $("#matrix-image").alt);
    $("#matrix-caption").textContent = `${tn + tp} correct predictions out of ${split.testing_records}. ${fp} false positives and ${fn} false negatives. Rows: actual labels; columns: predictions.`;
    $("#report-body").replaceChildren();
    for (const key of ["negative", "positive", "macro avg", "weighted avg"]) {
      const row = document.createElement("tr");
      const item = report[key];
      for (const value of [key[0].toUpperCase() + key.slice(1), `${percent(item.precision)}%`, `${percent(item.recall)}%`, `${percent(item["f1-score"])}%`, number(item.support)]) {
        const cell = document.createElement("td"); cell.textContent = value; row.append(cell);
      }
      $("#report-body").append(row);
    }
    $("#dataset-file").textContent = dataset.filename;
    $("#dataset-total").textContent = number(dataset.final_records);
    $("#positive-count").textContent = number(dataset.class_distribution.positive);
    $("#negative-count").textContent = number(dataset.class_distribution.negative);
    $("#positive-bar").style.width = `${100 * dataset.class_distribution.positive / dataset.final_records}%`;
    $("#distribution").setAttribute("aria-label", `${dataset.class_distribution.positive} positive and ${dataset.class_distribution.negative} negative reviews`);
    const facts = {"Original records": number(dataset.original_records), "Training / testing": `${number(split.training_records)} / ${number(split.testing_records)}`, "Missing values": number(Object.values(dataset.missing_values).reduce((a,b) => a+b, 0)), "Duplicate rows / review texts": `${dataset.duplicate_rows} / ${dataset.duplicate_review_texts}`, "Rows removed": number(dataset.removed_count)};
    $("#dataset-facts").replaceChildren();
    for (const [label, value] of Object.entries(facts)) {
      const div = document.createElement("div"); const dt = document.createElement("dt"); const dd = document.createElement("dd");
      dt.textContent = label; dd.textContent = value; div.append(dt,dd); $("#dataset-facts").append(div);
    }
    $("#feature-count").textContent = number(data.feature_count);
    $("#model-status").innerHTML = '<span class="dot"></span>Model ready';
    ready = true;
    button.disabled = false;
  } catch (error) {
    ready = false; button.disabled = true;
    $("#model-status").textContent = "Model unavailable";
    $("#load-error span").textContent = "Could not load the saved experiment. Check the local server and try again.";
    $("#load-error").hidden = false;
  }
}
$("#retry").addEventListener("click", loadExperiment);

function resetResult() {
  requestId += 1;
  if (controller) controller.abort();
  controller = null;
  button.disabled = !ready;
  $("#analyze-button span").textContent = "Analyze sentiment";
  $("#prediction").className = "prediction";
  $("#prediction").innerHTML = emptyResult;
  $("#input-error").hidden = true;
  review.removeAttribute("aria-invalid");
  $("#count").textContent = `${number(review.value.length)} / 20,000`;
  document.querySelectorAll("[data-example]").forEach((item) => {
    const selected = examples[item.dataset.example] === review.value;
    item.classList.toggle("selected", selected);
    item.setAttribute("aria-pressed", String(selected));
  });
}
review.addEventListener("input", resetResult);
$("#clear").addEventListener("click", () => {review.value = ""; resetResult(); review.focus();});
document.querySelectorAll("[data-example]").forEach((item) => item.addEventListener("click", () => {
  review.value = examples[item.dataset.example]; resetResult(); focusReview();
}));
review.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {event.preventDefault(); if (!button.disabled) $("#review-form").requestSubmit();}
});

function showError(message) {
  $("#input-error").textContent = message;
  $("#input-error").hidden = false;
  review.setAttribute("aria-invalid", "true");
  review.focus();
}

$("#review-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!ready || button.disabled) return;
  resetResult();
  if (!review.value.trim()) {showError("Write or paste a movie review before analyzing."); return;}
  const id = ++requestId;
  const activeController = new AbortController();
  controller = activeController;
  const timeout = setTimeout(() => activeController.abort(), 15000);
  button.disabled = true;
  $("#analyze-button span").textContent = "Analyzing…";
  $("#prediction").className = "prediction busy";
  $("#prediction h3").textContent = "Reading your review…";
  try {
    const response = await fetch("/api/predict", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({review:review.value}),signal:controller.signal});
    const result = await response.json();
    if (id !== requestId) return;
    if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "Please enter a review between 1 and 20,000 characters.");
    const positive = result.sentiment === "positive";
    $("#prediction").className = `prediction ${positive ? "positive" : "negative"}`;
    $("#prediction").innerHTML = `<div class="result-icon"><svg><use href="#${positive ? "positive-face" : "negative-face"}"/></svg></div><h3></h3><p></p><span class="result-placeholder">TF-IDF + Linear SVM</span>`;
    $("#prediction h3").textContent = result.label;
    $("#prediction p").textContent = positive ? "The model reads your review as positive." : "The model reads your review as negative.";
    // The result sits below the editor on phones; bring it into view after success.
    if (window.matchMedia("(max-width: 650px)").matches) {
      const verdict = $(".result-card");
      if (verdict.getBoundingClientRect().bottom > window.innerHeight) {
        review.blur();
        verdict.scrollIntoView({behavior: reducedMotion.matches ? "instant" : "smooth", block: "start"});
      }
    }
  } catch (error) {
    if (id !== requestId) return;
    $("#prediction").className = "prediction";
    $("#prediction").innerHTML = emptyResult;
    showError(error.name === "AbortError" ? "The request timed out. Please try again." : error.message === "Failed to fetch" ? "Cannot reach the local server. Check the connection and try again." : error.message);
  } finally {
    clearTimeout(timeout);
    if (id === requestId) {button.disabled = !ready; controller = null; $("#analyze-button span").textContent = "Analyze sentiment";}
  }
});
loadExperiment();
