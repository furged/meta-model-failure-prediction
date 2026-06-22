// Sentinel AI -- frontend logic.
// Talks to the real Flask API (/api/analyze, metrics from the page itself).
// No fabricated numbers anywhere in this file -- every value rendered
// here either comes from the server response or is a fixed UI constant
// (like the SVG circle's circumference).

const GAUGE_CIRCUMFERENCE = 2 * Math.PI * 52; // matches r=52 in the SVG

function $(id) {
  return document.getElementById(id);
}

// ---------------- Analyze flow ----------------

const form = $('analyze-form');
const textInput = $('text-input');
const runBtn = $('run-btn');
const runHint = $('run-hint');
const errorBox = $('error-box');
const resultBlock = $('result-block');

form.addEventListener('submit', async (e) => {
  e.preventDefault();

  const text = textInput.value.trim();

  if (!text) {
    showError('Please enter some text to analyze.');
    return;
  }

  setLoading(true);
  hideError();

  try {
    const res = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });

    const data = await res.json();

    if (!res.ok) {
      showError(data.error || 'Something went wrong. Please try again.');
      setLoading(false);
      return;
    }

    renderResult(data);
  } catch (err) {
    showError('Could not reach the server. Check your connection and try again.');
  } finally {
    setLoading(false);
  }
});

function setLoading(isLoading) {
  runBtn.disabled = isLoading;
  runBtn.textContent = isLoading ? 'ANALYZING...' : 'RUN ANALYSIS';
  runHint.textContent = isLoading ? 'running on cpu, hang tight' : '~1-3s on cpu';
  $('sys-status').textContent = isLoading ? 'BUSY' : 'READY';
  $('sys-status').className = isLoading ? 'status-warn' : 'status-ok';
}

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.hidden = false;
}

function hideError() {
  errorBox.hidden = true;
}

function renderResult(data) {
  resultBlock.hidden = false;

  // Verdict banner
  const banner = $('verdict-banner');
  const dot = $('verdict-dot');
  const text = $('verdict-text');
  const sub = $('verdict-sub');

  if (data.is_failure_risk) {
    banner.className = 'verdict-banner warn';
    dot.className = 'dot warn';
    text.className = 'verdict-text warn';
    text.textContent = data.warning;
  } else {
    banner.className = 'verdict-banner ok';
    dot.className = 'dot ok';
    text.className = 'verdict-text';
    text.textContent = data.warning;
  }

  sub.textContent =
    `failure probability ${data.failure_probability.toFixed(3)} \u00b7 threshold ${data.failure_threshold.toFixed(3)}`;

  // Gauge: animate stroke-dashoffset to reflect failure_probability (0-1)
  const gaugeFill = $('gauge-fill');
  const gaugeValue = $('gauge-value');
  const pct = Math.max(0, Math.min(1, data.failure_probability));
  const offset = GAUGE_CIRCUMFERENCE * (1 - pct);

  // Force reflow so the transition always plays, even if two analyses
  // in a row land on a similar percentage
  gaugeFill.style.transition = 'none';
  gaugeFill.style.strokeDashoffset = GAUGE_CIRCUMFERENCE;
  void gaugeFill.offsetWidth;
  gaugeFill.style.transition = '';
  gaugeFill.style.strokeDashoffset = offset;
  gaugeFill.style.stroke = data.is_failure_risk ? '#fbbf24' : '#4ade80';

  gaugeValue.textContent = `${Math.round(pct * 100)}%`;

  // Evidence rows
  $('ev-bert').textContent = `${data.bert_label} (${(data.bert_confidence * 100).toFixed(1)}% confident)`;
  $('ev-bert').className = 'v ' + (data.bert_label === 'positive' ? 'pos' : 'neg');

  $('ev-vader').textContent = data.vader_agrees
    ? `agrees \u00b7 ${data.vader_label}`
    : `disagrees \u00b7 ${data.vader_label}`;
  $('ev-vader').className = 'v ' + (data.vader_label === 'positive' ? 'pos' : 'neg');

  $('ev-lr').textContent = data.lr_agrees
    ? `agrees \u00b7 ${data.lr_label}`
    : `disagrees \u00b7 ${data.lr_label}`;
  $('ev-lr').className = 'v ' + (data.lr_label === 'positive' ? 'pos' : 'neg');

  $('ev-entropy').textContent = data.bert_entropy.toFixed(3);
  $('ev-entropy').className = 'v';

  resultBlock.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ---------------- Precision-Recall chart (real data, rendered once) ----------------

function renderPRChart() {
  const dataEl = $('metrics-data');
  const chartEl = $('pr-chart');

  if (!dataEl || !chartEl) return;

  let metrics;
  try {
    metrics = JSON.parse(dataEl.textContent);
  } catch (e) {
    return;
  }

  if (!metrics || !metrics.pr_curve || metrics.pr_curve.length === 0) return;

  const points = metrics.pr_curve;
  const width = 400;
  const height = 260;
  const padding = { top: 12, right: 16, bottom: 36, left: 48 };
  const plotW = width - padding.left - padding.right;
  const plotH = height - padding.top - padding.bottom;

  function xPos(recall) {
    return padding.left + recall * plotW;
  }
  function yPos(precision) {
    return padding.top + (1 - precision) * plotH;
  }

  // Build the path for the PR curve, sorted by recall ascending so the
  // line draws left-to-right without crossing back on itself.
  const sorted = [...points].sort((a, b) => a.recall - b.recall);
  const pathD = sorted
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${xPos(p.recall).toFixed(1)} ${yPos(p.precision).toFixed(1)}`)
    .join(' ');

  // Axis lines + ticks
  let svg = '';

  // Gridlines at 0.25 / 0.5 / 0.75 / 1.0
  [0, 0.25, 0.5, 0.75, 1].forEach((t) => {
    const y = yPos(t);
    svg += `<line x1="${padding.left}" y1="${y.toFixed(1)}" x2="${width - padding.right}" y2="${y.toFixed(1)}" stroke="#1a1a1a" stroke-width="1" />`;
    svg += `<text x="${padding.left - 8}" y="${(y + 3).toFixed(1)}" text-anchor="end" font-size="11" fill="#6b6b6b" font-family="JetBrains Mono, monospace">${t}</text>`;
  });

  [0, 0.25, 0.5, 0.75, 1].forEach((t) => {
    const x = xPos(t);
    svg += `<text x="${x.toFixed(1)}" y="${height - padding.bottom + 18}" text-anchor="middle" font-size="11" fill="#6b6b6b" font-family="JetBrains Mono, monospace">${t}</text>`;
  });

  svg += `<text x="${padding.left + plotW / 2}" y="${height - 4}" text-anchor="middle" font-size="12" fill="#4a4a4a" font-family="JetBrains Mono, monospace">recall</text>`;
  svg += `<text x="14" y="${padding.top + plotH / 2}" text-anchor="middle" font-size="12" fill="#4a4a4a" font-family="JetBrains Mono, monospace" transform="rotate(-90 14 ${padding.top + plotH / 2})">precision</text>`;

  // The curve itself
  svg += `<path d="${pathD}" fill="none" stroke="#4ade80" stroke-width="0.75" />`;

  // Mark the operating point at the chosen threshold. Rather than using
  // metrics.precision/metrics.recall directly (computed separately during
  // training, via predict>=threshold on the test set), find the closest
  // point that's actually ON the rendered curve -- the two calculations
  // are mathematically related but not numerically identical, so using
  // the raw threshold values could place the marker slightly off the
  // line. Snapping to the nearest real curve point guarantees it sits
  // exactly on the line.
  let closest = sorted[0];
  let closestDist = Infinity;
  for (const p of sorted) {
    const d = Math.abs(p.recall - metrics.recall) + Math.abs(p.precision - metrics.precision);
    if (d < closestDist) {
      closestDist = d;
      closest = p;
    }
  }
  const opX = xPos(closest.recall);
  const opY = yPos(closest.precision);

  svg += `<circle cx="${opX.toFixed(1)}" cy="${opY.toFixed(1)}" r="3" fill="#fbbf24" />`;

  // Corner legend instead of a floating label on the curve -- a text
  // label tied to the marker's position would overlap the line or run
  // off the edge depending on where the threshold happens to fall.
  // A fixed-position legend in the corner is always readable.
  const legendX = width - padding.right - 14;
  const legendY = padding.top + 4;
  svg += `<circle cx="${legendX}" cy="${legendY}" r="3" fill="#fbbf24" />`;
  svg += `<text x="${legendX - 8}" y="${(legendY + 3.5).toFixed(1)}" text-anchor="end" font-size="10" fill="#fbbf24" font-family="JetBrains Mono, monospace">threshold ${metrics.threshold.toFixed(3)}</text>`;

  chartEl.innerHTML = svg;
}

renderPRChart();

// ---------------- Tab navigation ----------------

const tabButtons = document.querySelectorAll('.tab-btn');
const tabPanels = {
  analyze: $('tab-analyze'),
  diagnostics: $('tab-diagnostics'),
  ping: $('tab-ping')
};

tabButtons.forEach((btn) => {
  btn.addEventListener('click', () => {
    const target = btn.dataset.tab;

    tabButtons.forEach((b) => {
      const isActive = b === btn;
      b.classList.toggle('active', isActive);
      b.setAttribute('aria-selected', isActive ? 'true' : 'false');
    });

    Object.entries(tabPanels).forEach(([name, panel]) => {
      if (!panel) return;
      panel.hidden = name !== target;
    });
  });
});

// ---------------- Feedback form ----------------

const feedbackForm = $('feedback-form');

if (feedbackForm) {
  const feedbackEmail = $('feedback-email');
  const feedbackMessage = $('feedback-message');
  const feedbackBtn = $('feedback-btn');
  const feedbackHint = $('feedback-hint');
  const feedbackErrorBox = $('feedback-error-box');
  const feedbackSuccessBox = $('feedback-success-box');

  feedbackForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const email = feedbackEmail.value.trim();
    const message = feedbackMessage.value.trim();

    feedbackErrorBox.hidden = true;
    feedbackSuccessBox.hidden = true;

    if (!email) {
      showFeedbackError('Please enter your email.');
      return;
    }

    if (!message) {
      showFeedbackError('Please enter a message.');
      return;
    }

    feedbackBtn.disabled = true;
    feedbackBtn.textContent = 'SENDING...';
    feedbackHint.textContent = '';

    try {
      const res = await fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, message })
      });

      const data = await res.json();

      if (!res.ok) {
        showFeedbackError(data.error || 'Something went wrong. Please try again.');
        return;
      }

      feedbackSuccessBox.hidden = false;
      feedbackForm.reset();
    } catch (err) {
      showFeedbackError('Could not reach the server. Check your connection and try again.');
    } finally {
      feedbackBtn.disabled = false;
      feedbackBtn.textContent = 'SEND';
    }
  });

  function showFeedbackError(msg) {
    feedbackErrorBox.textContent = msg;
    feedbackErrorBox.hidden = false;
  }
}
