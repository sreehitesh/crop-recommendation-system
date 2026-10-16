/* ==========================================================================
   Crop Recommendation System - frontend behaviour
   - responsive sidebar
   - Chart.js charts with a plain-HTML fallback when the CDN is unavailable
   - prediction form: client-side validation mirroring the server rules,
     POST /api/predict, and rendering of the result card
   ========================================================================== */
(function () {
  "use strict";

  var PAGE_DATA = window.__PAGE_DATA__ || {};
  var APP = window.__APP__ || {};
  var CHART_COLORS = ["#1b5e20", "#43a047", "#9ccc65", "#2e7d32", "#66bb6a", "#a5d6a7"];

  /* ------------------------------------------------------------- utilities */
  function $(selector, scope) {
    return (scope || document).querySelector(selector);
  }

  function $all(selector, scope) {
    return Array.prototype.slice.call((scope || document).querySelectorAll(selector));
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function formatNumber(value, digits) {
    if (value === null || value === undefined || isNaN(value)) return "–";
    return Number(value).toFixed(digits === undefined ? 4 : digits);
  }

  function chartAvailable() {
    return typeof window.Chart !== "undefined";
  }

  /* --------------------------------------------------------------- sidebar */
  function initSidebar() {
    var toggle = $("#menuToggle");
    var sidebar = $("#sidebar");
    var scrim = $("#sidebarScrim");
    if (!toggle || !sidebar) return;

    function setOpen(open) {
      sidebar.classList.toggle("is-open", open);
      toggle.setAttribute("aria-expanded", String(open));
      if (scrim) scrim.hidden = !open;
    }

    toggle.addEventListener("click", function () {
      setOpen(!sidebar.classList.contains("is-open"));
    });
    if (scrim) scrim.addEventListener("click", function () { setOpen(false); });
    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape") setOpen(false);
    });
    $all(".nav-link", sidebar).forEach(function (link) {
      link.addEventListener("click", function () { setOpen(false); });
    });
  }

  /* ---------------------------------------------------------------- charts */
  var chartRegistry = typeof WeakMap !== "undefined" ? new WeakMap() : null;

  /**
   * Render a chart, or a readable HTML bar list when Chart.js is unavailable.
   * data: { labels: [...], series: [{ name, values, color }], horizontal?: bool }
   *
   * Chart.js v4 needs a real <canvas> element, so one is created inside the
   * container on every render and any previous instance is destroyed first.
   */
  function renderChart(container, data) {
    if (!container) return;

    if (chartRegistry) {
      var previous = chartRegistry.get(container);
      if (previous) {
        previous.destroy();
        chartRegistry.delete(container);
      }
    }

    if (chartAvailable()) {
      container.innerHTML = "";
      var canvas = document.createElement("canvas");
      canvas.setAttribute("role", "img");
      canvas.setAttribute("aria-label", (data.ariaLabel || "Chart") + " - a text summary is available in the tables on this page.");
      container.appendChild(canvas);
      try {
        var chart = new window.Chart(canvas, {
          type: data.type || "bar",
          data: {
            labels: data.labels,
            datasets: data.series.map(function (series, index) {
              return {
                label: series.name,
                data: series.values,
                backgroundColor: series.color || (data.type === "line" ? "#43a047" : CHART_COLORS[index % CHART_COLORS.length]),
                borderColor: series.color || "#1b5e20",
                borderWidth: data.type === "line" ? 2 : 1,
                borderRadius: data.type === "bar" ? 6 : 0,
                tension: 0.3,
                fill: false
              };
            })
          },
          options: {
            indexAxis: data.horizontal ? "y" : "x",
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: { display: data.series.length > 1, position: "bottom", labels: { boxWidth: 12, usePointStyle: true } },
              tooltip: {
                callbacks: {
                  label: function (context) {
                    var value = context.parsed[data.horizontal ? "x" : "y"];
                    return " " + context.dataset.label + ": " + (data.decimals === 0 ? value : Number(value).toFixed(data.decimals === undefined ? 3 : data.decimals)) + (data.suffix || "");
                  }
                }
              }
            },
            scales: {
              x: { beginAtZero: true, grid: { color: "#eef3ee" }, ticks: { autoSkip: false, maxRotation: data.type === "bar" && !data.horizontal ? 90 : 0, minRotation: 0, font: { size: 10 } } },
              y: { beginAtZero: true, grid: { color: "#eef3ee" }, ticks: { font: { size: 10 } } }
            }
          }
        });
        if (chartRegistry) chartRegistry.set(container, chart);
        return;
      } catch (error) {
        container.innerHTML = "";
        /* fall through to the HTML fallback */
      }
    }
    renderFallbackBars(container, data);
  }

  function renderFallbackBars(container, data) {
    var primary = data.series[0] || { values: [] };
    var max = Math.max.apply(null, primary.values.concat([1]));
    var html = ['<div class="chart-fallback">'];
    html.push('<p class="muted" style="font-size:.78rem">Interactive charts need Chart.js from the CDN; showing a static summary instead.</p>');
    data.labels.forEach(function (label, index) {
      var value = primary.values[index];
      var pct = Math.max(2, Math.round((value / max) * 100));
      html.push(
        '<div class="chart-fallback-row"><span>' + escapeHtml(label) + "</span>" +
          '<span class="chart-fallback-bar"><span style="width:' + pct + '%"></span></span>' +
          "<span>" + (data.decimals === 0 ? value : formatNumber(value, data.decimals === undefined ? 3 : data.decimals)) + "</span></div>"
      );
    });
    html.push("</div>");
    container.innerHTML = html.join("");
  }

  function classDistributionData(counts) {
    var labels = Object.keys(counts);
    return {
      labels: labels,
      series: [{ name: "Records", values: labels.map(function (k) { return counts[k]; }), color: "#2e7d32" }],
      decimals: 0
    };
  }

  function initOverviewCharts() {
    var container = $('[data-chart="class-distribution"]');
    if (!container || !PAGE_DATA.classCounts) return;
    renderChart(container, classDistributionData(PAGE_DATA.classCounts));
  }

  function initDatasetCharts() {
    var distribution = $('[data-chart="class-distribution"]');
    if (distribution && PAGE_DATA.classCounts) {
      renderChart(distribution, classDistributionData(PAGE_DATA.classCounts));
    }

    var histogram = $('[data-chart="feature-histogram"]');
    var select = $("#featureSelect");
    if (!histogram || !select || !PAGE_DATA.histograms) return;

    function drawHistogram() {
      var feature = select.value;
      var data = PAGE_DATA.histograms[feature];
      if (!data) return;
      renderChart(histogram, {
        labels: data.labels,
        series: [{ name: "Records", values: data.counts, color: "#66bb6a" }],
        decimals: 0,
        ariaLabel: "Histogram of " + feature
      });
    }

    select.addEventListener("change", drawHistogram);
    drawHistogram();
  }

  function initPerformanceCharts() {
    var comparison = $('[data-chart="model-comparison"]');
    if (comparison && PAGE_DATA.comparison) {
      renderChart(comparison, {
        labels: PAGE_DATA.comparison.labels,
        series: PAGE_DATA.comparison.models.map(function (model, index) {
          return {
            name: model.name + (model.selected ? " (selected)" : ""),
            values: model.values,
            color: model.selected ? "#1b5e20" : CHART_COLORS[(index + 1) % CHART_COLORS.length]
          };
        }),
        decimals: 3
      });
    }

    var importance = $('[data-chart="feature-importance"]');
    if (importance && PAGE_DATA.featureImportance) {
      var entries = Object.keys(PAGE_DATA.featureImportance).map(function (key) {
        return [key, PAGE_DATA.featureImportance[key]];
      }).sort(function (a, b) { return a[1] - b[1]; });
      renderChart(importance, {
        labels: entries.map(function (entry) { return entry[0]; }),
        series: [{ name: "Relative importance", values: entries.map(function (entry) { return entry[1]; }), color: "#2e7d32" }],
        horizontal: true,
        decimals: 3
      });
    }

    // Confusion matrix tabs
    var image = $("#confusionImage");
    var tabs = $all(".tab-btn[data-matrix]");
    tabs.forEach(function (tab) {
      tab.addEventListener("click", function () {
        tabs.forEach(function (other) { other.classList.remove("is-active"); });
        tab.classList.add("is-active");
        if (image) {
          image.src = "/reports/" + tab.getAttribute("data-matrix") + ".png";
          image.alt = "Confusion matrix: " + tab.textContent.trim();
        }
      });
    });
  }

  /* ------------------------------------------------------------- tooltips */
  function tip(message, type) {
    return '<div class="alert alert-' + (type || "error") + '">' + escapeHtml(message) + "</div>";
  }

  /* -------------------------------------------------------- prediction form */
  function initPredictionForm() {
    var form = $("#predictForm");
    if (!form) return;

    var features = PAGE_DATA.features || {};
    var example = PAGE_DATA.example || {};
    var fieldNames = Object.keys(features);
    var resultCard = $("#resultCard");
    var resultError = $("#resultError");
    var resultHint = $("#resultHint");
    var placeholder = $("#resultPlaceholder");

    function clearFieldErrors() {
      fieldNames.forEach(function (name) {
        var wrap = $('[data-field-wrap="' + name + '"]');
        if (wrap) wrap.classList.remove("has-error");
        var error = $("#err-" + name);
        if (error) { error.textContent = ""; }
      });
    }

    function showFieldError(name, message) {
      var wrap = $('[data-field-wrap="' + name + '"]');
      if (wrap) wrap.classList.add("has-error");
      var error = $("#err-" + name);
      if (error) error.textContent = message;
    }

    /** Client-side mirror of src/validate_data.py hard rules. */
    function validateLocally(values) {
      var errors = {};
      var firstInvalid = null;
      fieldNames.forEach(function (name) {
        var spec = features[name];
        var input = $('#input-' + name);
        var raw = values[name];
        var trimmed = raw === undefined || raw === null ? "" : String(raw).trim();

        if (trimmed === "") {
          errors[name] = spec.label + " is required.";
        } else if (!/^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$/.test(trimmed)) {
          errors[name] = spec.label + " must be a number.";
        } else {
          var numeric = Number(trimmed);
          if (!isFinite(numeric)) {
            errors[name] = spec.label + " must be a finite number.";
          } else if (numeric < spec.hard_min || numeric > spec.hard_max) {
            errors[name] = spec.label + " must be between " + spec.hard_min + " and " + spec.hard_max + ".";
          }
        }
        if (errors[name] && !firstInvalid) firstInvalid = input;
      });
      return { errors: errors, firstInvalid: firstInvalid };
    }

    function collectValues() {
      var values = {};
      fieldNames.forEach(function (name) {
        var input = $('#input-' + name);
        values[name] = input ? input.value : "";
      });
      return values;
    }

    function renderResult(payload) {
      var info = payload.crop_info || {};
      var rows = Object.keys(payload.input_values || {}).map(function (name) {
        var unit = features[name] ? features[name].unit : "";
        return "<div><dt>" + escapeHtml(features[name] ? features[name].label : name) + "</dt><dd>" +
          formatNumber(payload.input_values[name], 4) + "</dd></div>";
      }).join("");

      var warnings = (payload.warnings || []).length
        ? '<div class="alert alert-warn" style="margin-top:1rem">' +
            "<div><strong>Warnings</strong><ul>" +
            payload.warnings.map(function (w) { return "<li>" + escapeHtml(w) + "</li>"; }).join("") +
            "</ul></div></div>"
        : "";

      var scores = "";
      if (payload.scores_available && (payload.top_scores || []).length) {
        scores =
          '<h3 style="margin-top:1.1rem">Model scores for the top classes</h3>' +
          '<div class="score-list">' +
          payload.top_scores.map(function (entry) {
            var pct = Math.max(2, Math.round(entry.score * 100));
            return '<div class="score-row"><span>' + escapeHtml(entry.crop) + '</span>' +
              '<span class="score-track"><span style="width:' + pct + '%"></span></span>' +
              '<span class="score-value">' + formatNumber(entry.score, 4) + "</span></div>";
          }).join("") +
          "</div>" +
          '<p class="muted" style="font-size:.8rem">Each score is the ' + escapeHtml(payload.scores_meaning || "model score for that class") + "</p>";
      }

      resultCard.innerHTML =
        '<div class="result-hero" style="margin-bottom:.6rem">' +
          '<div class="result-crop">' +
            '<span class="result-crop-mark">' +
              '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">' +
              '<path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/>' +
              '<path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/></svg>' +
            "</span>" +
            "<div><div class=\"muted\" style=\"font-size:.76rem;text-transform:uppercase;letter-spacing:.08em\">Predicted crop</div>" +
            '<div class="result-crop-name">' + escapeHtml(payload.predicted_crop) + "</div></div>" +
          "</div>" +
          '<div class="result-meta">' +
            '<span class="pill">' + escapeHtml(info.category || "Crop") + "</span>" +
            '<span class="pill">Model: ' + escapeHtml(payload.model_name || "unknown") + "</span>" +
          "</div>" +
        "</div>" +
        (info.note ? '<p style="margin:.2rem 0 .8rem">' + escapeHtml(info.note) + "</p>" : "") +
        warnings +
        '<h3 style="margin-top:1rem">Values you submitted</h3>' +
        '<dl class="kv">' + rows + "</dl>" +
        scores +
        '<div class="alert alert-info" style="margin:1rem 0 .8rem">' +
          "<div><strong>Please note.</strong> This is a machine learning estimate based on the dataset's " +
          "patterns. It is not a guarantee of crop success and should be checked against local soil " +
          "tests, season and agricultural guidance.</div>" +
        "</div>" +
        '<div class="btn-row">' +
          '<button type="button" class="btn btn-outline" id="anotherBtn">Make another prediction</button>' +
          '<a class="btn btn-ghost" href="/performance">See how the model was evaluated</a>' +
        "</div>";

      resultError.hidden = true;
      resultError.innerHTML = "";
      if (resultHint) resultHint.hidden = true;
      resultCard.hidden = false;
      if (placeholder) placeholder.scrollIntoView({ behavior: "smooth", block: "nearest" });

      var another = $("#anotherBtn");
      if (another) {
        another.addEventListener("click", function () {
          resultCard.hidden = true;
          resultCard.innerHTML = "";
          if (resultHint) resultHint.hidden = false;
          var first = $("#input-" + fieldNames[0]);
          if (first) first.focus();
        });
      }
    }

    form.addEventListener("submit", function (event) {
      event.preventDefault();
      clearFieldErrors();

      var values = collectValues();
      var local = validateLocally(values);
      if (Object.keys(local.errors).length) {
        Object.keys(local.errors).forEach(function (name) { showFieldError(name, local.errors[name]); });
        resultCard.hidden = true;
        if (resultHint) resultHint.hidden = true;
        resultError.hidden = false;
        resultError.innerHTML = tip("Please correct the highlighted fields before predicting.");
        if (local.firstInvalid) local.firstInvalid.focus();
        return;
      }

      var button = $("#predictBtn");
      var originalLabel = button ? button.innerHTML : "";
      if (button) { button.disabled = true; button.innerHTML = "Predicting…"; }
      resultError.hidden = true;
      resultError.innerHTML = "";

      fetch(APP.predictUrl || "/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(values)
      })
        .then(function (response) {
          return response.json().catch(function () {
            return { status: "error", error: "The server returned an unreadable response (" + response.status + ")." };
          }).then(function (body) {
            return { ok: response.ok, status: response.status, body: body };
          });
        })
        .then(function (result) {
          if (result.ok && result.body.status === "success") {
            renderResult(result.body);
            return;
          }
          var body = result.body || {};
          if (body.field_errors) {
            Object.keys(body.field_errors).forEach(function (name) { showFieldError(name, body.field_errors[name]); });
          }
          resultCard.hidden = true;
          if (resultHint) resultHint.hidden = true;
          resultError.hidden = false;
          var extra = body.train_command ? '<p style="margin:.4rem 0 0">Train the model with <code>' + escapeHtml(body.train_command) + "</code>.</p>" : "";
          resultError.innerHTML = tip(body.error || "The prediction could not be completed.") + extra;
        })
        .catch(function () {
          resultCard.hidden = true;
          if (resultHint) resultHint.hidden = true;
          resultError.hidden = false;
          resultError.innerHTML = tip("Could not reach the server. Is the Flask application still running?");
        })
        .finally(function () {
          if (button) { button.disabled = false; button.innerHTML = originalLabel; }
        });
    });

    var exampleBtn = $("#exampleBtn");
    if (exampleBtn) {
      exampleBtn.addEventListener("click", function () {
        fieldNames.forEach(function (name) {
          var input = $('#input-' + name);
          if (input && example[name] !== undefined) input.value = example[name];
        });
        clearFieldErrors();
        if (resultError) { resultError.hidden = true; resultError.innerHTML = ""; }
        // Deliberately NO prediction here: the button only fills the form.
      });
    }

    var resetBtn = $("#resetBtn");
    if (resetBtn) {
      resetBtn.addEventListener("click", function () {
        form.reset();
        clearFieldErrors();
        if (resultCard) { resultCard.hidden = true; resultCard.innerHTML = ""; }
        if (resultError) { resultError.hidden = true; resultError.innerHTML = ""; }
        if (resultHint) resultHint.hidden = false;
        var first = $("#input-" + fieldNames[0]);
        if (first) first.focus();
      });
    }

    // Clear a field error as soon as the user edits that field again.
    fieldNames.forEach(function (name) {
      var input = $('#input-' + name);
      if (!input) return;
      input.addEventListener("input", function () {
        var wrap = $('[data-field-wrap="' + name + '"]');
        if (wrap) wrap.classList.remove("has-error");
        var error = $("#err-" + name);
        if (error) error.textContent = "";
      });
    });
  }

  /* -------------------------------------------------------------- bootstrap */
  document.addEventListener("DOMContentLoaded", function () {
    initSidebar();
    initOverviewCharts();
    initDatasetCharts();
    initPerformanceCharts();
    initPredictionForm();
  });
})();
