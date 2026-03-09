/**
 * LMT Experiment Result Charts
 *
 * Interactive Canvas-based bar charts for experiment results.
 * Designed for the LMT documentation site (mkdocs-material).
 *
 * Usage: Add a <div class="exp-chart" data-experiment="arch-comparison"></div>
 * to any markdown page.
 */
(function () {
  'use strict';

  // --- Theme Detection ---

  function isDarkTheme() {
    var body = document.body;
    return body.getAttribute('data-md-color-scheme') === 'slate';
  }

  function getThemeColors() {
    var dark = isDarkTheme();
    return {
      bg: dark ? '#1e1e2e' : '#ffffff',
      text: dark ? '#cdd6f4' : '#1a1a2e',
      textMuted: dark ? '#6c7086' : '#888888',
      gridLine: dark ? '#45475a' : '#e0e0e0',
      barColors: dark
        ? ['#cba6f7', '#89b4fa', '#a6e3a1', '#f9e2af', '#f38ba8']
        : ['#6a1b9a', '#1565c0', '#2e7d32', '#f57f17', '#c62828'],
      tooltipBg: dark ? '#313244' : '#ffffff',
      tooltipBorder: dark ? '#585b70' : '#cccccc',
    };
  }

  // --- Data ---

  var EXPERIMENTS = {
    'arch-comparison-random': {
      title: 'Architecture Comparison — Random Data',
      ylabel: 'Best BPB',
      baseline: 8.0,
      data: [
        { label: 'LLaMA', value: 8.0300 },
        { label: 'Qwen3', value: 8.0300 },
        { label: 'Gemma', value: 8.0309 },
        { label: 'Mixtral', value: 8.0309 },
        { label: 'GPT', value: 8.0322 },
      ],
    },
    'arch-comparison-wikitext': {
      title: 'Architecture Comparison — WikiText-2',
      ylabel: 'Best BPB (lower is better)',
      data: [
        { label: 'Mixtral', value: 3.014 },
        { label: 'LLaMA', value: 3.167 },
        { label: 'Qwen3', value: 3.167 },
        { label: 'Gemma', value: 3.376 },
        { label: 'GPT', value: 3.556 },
      ],
    },
    'arch-ablation': {
      title: 'Architecture Ablation — WikiText-2',
      ylabel: 'Best BPB (lower is better)',
      data: [
        { label: 'LLaMA\n(all 4)', value: 3.226 },
        { label: 'GPT+GQA', value: 3.583 },
        { label: 'GPT+RoPE', value: 3.587 },
        { label: 'GPT', value: 3.602 },
        { label: 'GPT+RMSNorm', value: 3.607 },
        { label: 'GPT+SwiGLU', value: 3.617 },
      ],
    },
    'pairwise-ablation': {
      title: 'Pairwise Ablation — WikiText-2',
      ylabel: 'Best BPB (lower is better)',
      data: [
        { label: 'RoPE+GQA', value: 3.140 },
        { label: 'LLaMA\n(full)', value: 3.226 },
        { label: 'GQA+\nRMSNorm', value: 3.585 },
        { label: 'RoPE+\nRMSNorm', value: 3.594 },
        { label: 'SwiGLU+\nGQA', value: 3.596 },
        { label: 'GPT', value: 3.602 },
        { label: 'RoPE+\nSwiGLU', value: 3.613 },
        { label: 'SwiGLU+\nRMSNorm', value: 3.619 },
      ],
    },
    'babylm-results': {
      title: 'BabyLM Pretraining — BPB',
      ylabel: 'BPB',
      data: [
        { label: 'Trained GPT', value: 2.41 },
        { label: 'Random', value: 10.0 },
      ],
    },
  };

  // --- Rendering ---

  function drawChart(container) {
    var expKey = container.getAttribute('data-experiment');
    var exp = EXPERIMENTS[expKey];
    if (!exp) {
      container.textContent = 'Unknown experiment: ' + expKey;
      return;
    }

    var colors = getThemeColors();
    var dpr = window.devicePixelRatio || 1;

    // Sizing
    var width = Math.min(container.clientWidth || 600, 700);
    var height = 320;
    var margin = { top: 40, right: 20, bottom: 50, left: 60 };
    var plotW = width - margin.left - margin.right;
    var plotH = height - margin.top - margin.bottom;

    // Create canvas
    var canvas = document.createElement('canvas');
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';
    container.innerHTML = '';
    container.appendChild(canvas);

    var ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    // Background
    ctx.fillStyle = colors.bg;
    ctx.fillRect(0, 0, width, height);

    // Title
    ctx.fillStyle = colors.text;
    ctx.font = 'bold 14px -apple-system, BlinkMacSystemFont, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(exp.title, width / 2, 24);

    // Compute scale
    var values = exp.data.map(function (d) { return d.value; });
    var minVal = Math.min.apply(null, values);
    var maxVal = Math.max.apply(null, values);
    var range = maxVal - minVal;
    var yMin = Math.max(0, minVal - range * 0.3);
    var yMax = maxVal + range * 0.15;
    if (range < 0.01) {
      // Very small range (like random data) — zoom in
      yMin = minVal - 0.005;
      yMax = maxVal + 0.005;
    }

    var barCount = exp.data.length;
    var barGap = 8;
    var barW = Math.min(60, (plotW - barGap * (barCount + 1)) / barCount);
    var totalBarsW = barCount * barW + (barCount - 1) * barGap;
    var startX = margin.left + (plotW - totalBarsW) / 2;

    // Y-axis gridlines
    var yTicks = 5;
    ctx.font = '11px -apple-system, BlinkMacSystemFont, monospace';
    ctx.textAlign = 'right';
    for (var i = 0; i <= yTicks; i++) {
      var yVal = yMin + (yMax - yMin) * (1 - i / yTicks);
      var yPos = margin.top + (i / yTicks) * plotH;

      ctx.strokeStyle = colors.gridLine;
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      ctx.moveTo(margin.left, yPos);
      ctx.lineTo(width - margin.right, yPos);
      ctx.stroke();

      ctx.fillStyle = colors.textMuted;
      ctx.fillText(yVal.toFixed(3), margin.left - 6, yPos + 4);
    }

    // Y-axis label
    ctx.save();
    ctx.translate(14, margin.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = colors.textMuted;
    ctx.font = '11px -apple-system, BlinkMacSystemFont, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(exp.ylabel, 0, 0);
    ctx.restore();

    // Bars
    for (var j = 0; j < barCount; j++) {
      var d = exp.data[j];
      var barX = startX + j * (barW + barGap);
      var barH = ((d.value - yMin) / (yMax - yMin)) * plotH;
      var barY = margin.top + plotH - barH;

      // Bar fill
      ctx.fillStyle = colors.barColors[j % colors.barColors.length];
      ctx.fillRect(barX, barY, barW, barH);

      // Value label on bar
      ctx.fillStyle = colors.text;
      ctx.font = 'bold 11px -apple-system, BlinkMacSystemFont, monospace';
      ctx.textAlign = 'center';
      ctx.fillText(d.value.toFixed(3), barX + barW / 2, barY - 6);

      // X-axis label
      ctx.fillStyle = colors.text;
      ctx.font = '12px -apple-system, BlinkMacSystemFont, sans-serif';
      ctx.fillText(d.label, barX + barW / 2, margin.top + plotH + 18);
    }

    // Baseline line (if present)
    if (exp.baseline !== undefined) {
      var baseY = margin.top + plotH -
        ((exp.baseline - yMin) / (yMax - yMin)) * plotH;
      ctx.strokeStyle = colors.barColors[0];
      ctx.lineWidth = 1.5;
      ctx.setLineDash([5, 3]);
      ctx.beginPath();
      ctx.moveTo(margin.left, baseY);
      ctx.lineTo(width - margin.right, baseY);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = colors.textMuted;
      ctx.font = '10px -apple-system, BlinkMacSystemFont, sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(
        'theoretical optimum: ' + exp.baseline.toFixed(1),
        margin.left + 4, baseY - 6
      );
    }
  }

  // --- Init ---

  function initCharts() {
    var containers = document.querySelectorAll('.exp-chart');
    for (var i = 0; i < containers.length; i++) {
      drawChart(containers[i]);
    }
  }

  // Re-render on theme change
  var observer = new MutationObserver(function (mutations) {
    for (var i = 0; i < mutations.length; i++) {
      if (mutations[i].attributeName === 'data-md-color-scheme') {
        initCharts();
        break;
      }
    }
  });

  if (document.body) {
    observer.observe(document.body, { attributes: true });
  }

  // Init on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initCharts);
  } else {
    initCharts();
  }

  // Re-init on navigation (mkdocs-material instant loading)
  if (typeof document$ !== 'undefined') {
    document$.subscribe(initCharts);
  }
})();
