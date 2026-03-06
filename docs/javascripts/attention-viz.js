/**
 * LMT Attention Pattern Explorer
 *
 * Interactive Canvas-based visualization of attention mask patterns.
 * Designed for the LMT documentation site (mkdocs-material).
 *
 * Usage: Add a <div class="attn-viz" data-seq-len="12"></div> to any
 * markdown page, and this script renders an interactive heatmap.
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
      highlight: dark ? '#f5c2e7' : '#6a1b9a',
      hotColor: dark ? [203, 166, 247] : [106, 27, 154],   // purple
      coldColor: dark ? [30, 30, 46] : [245, 245, 255],     // near-bg
      controlBg: dark ? '#313244' : '#f5f5f5',
      controlBorder: dark ? '#45475a' : '#cccccc',
      controlActive: dark ? '#cba6f7' : '#6a1b9a',
      tooltipBg: dark ? '#313244' : '#ffffff',
      tooltipBorder: dark ? '#585b70' : '#cccccc',
    };
  }

  // --- Attention Pattern Generators ---

  function causalPattern(n) {
    var m = [];
    for (var i = 0; i < n; i++) {
      var row = [];
      for (var j = 0; j < n; j++) {
        row.push(j <= i ? 1.0 : 0.0);
      }
      m.push(row);
    }
    return m;
  }

  function slidingWindowPattern(n, windowSize) {
    var m = [];
    for (var i = 0; i < n; i++) {
      var row = [];
      for (var j = 0; j < n; j++) {
        var inWindow = j <= i && (i - j) < windowSize;
        row.push(inWindow ? 1.0 : 0.0);
      }
      m.push(row);
    }
    return m;
  }

  function stridedPattern(n, stride) {
    var m = [];
    for (var i = 0; i < n; i++) {
      var row = [];
      for (var j = 0; j < n; j++) {
        var causal = j <= i;
        var onStride = (i - j) % stride === 0;
        row.push(causal && onStride ? 1.0 : 0.0);
      }
      m.push(row);
    }
    return m;
  }

  function globalLocalPattern(n, windowSize, globalTokens) {
    var m = [];
    for (var i = 0; i < n; i++) {
      var row = [];
      for (var j = 0; j < n; j++) {
        var causal = j <= i;
        var inWindow = (i - j) < windowSize;
        var isGlobal = j < globalTokens || i < globalTokens;
        row.push(causal && (inWindow || isGlobal) ? 1.0 : 0.0);
      }
      m.push(row);
    }
    return m;
  }

  function fullPattern(n) {
    var m = [];
    for (var i = 0; i < n; i++) {
      var row = [];
      for (var j = 0; j < n; j++) {
        row.push(1.0);
      }
      m.push(row);
    }
    return m;
  }

  function softmaxRow(row) {
    var maxVal = -Infinity;
    for (var i = 0; i < row.length; i++) {
      if (row[i] > maxVal) maxVal = row[i];
    }
    var expSum = 0;
    var exps = [];
    for (var i = 0; i < row.length; i++) {
      if (row[i] === 0.0) {
        exps.push(0);
      } else {
        var e = Math.exp(row[i] - maxVal);
        exps.push(e);
        expSum += e;
      }
    }
    var result = [];
    for (var i = 0; i < exps.length; i++) {
      result.push(expSum > 0 ? exps[i] / expSum : 0);
    }
    return result;
  }

  function applySoftmax(matrix) {
    return matrix.map(softmaxRow);
  }

  // --- Rendering ---

  function renderHeatmap(canvas, matrix, colors, hoverCell) {
    var ctx = canvas.getContext('2d');
    var n = matrix.length;
    var dpr = window.devicePixelRatio || 1;

    var containerWidth = canvas.parentElement.clientWidth;
    var maxCellSize = 40;
    var minCellSize = 16;
    var labelMargin = 36;

    var availableSize = Math.min(containerWidth - labelMargin - 20, 600);
    var cellSize = Math.max(minCellSize, Math.min(maxCellSize,
      Math.floor(availableSize / n)));
    var gridSize = cellSize * n;

    canvas.style.width = (gridSize + labelMargin) + 'px';
    canvas.style.height = (gridSize + labelMargin) + 'px';
    canvas.width = (gridSize + labelMargin) * dpr;
    canvas.height = (gridSize + labelMargin) * dpr;
    ctx.scale(dpr, dpr);

    // Background
    ctx.fillStyle = colors.bg;
    ctx.fillRect(0, 0, gridSize + labelMargin, gridSize + labelMargin);

    // Cells
    for (var i = 0; i < n; i++) {
      for (var j = 0; j < n; j++) {
        var val = matrix[i][j];
        var x = labelMargin + j * cellSize;
        var y = labelMargin + i * cellSize;

        // Interpolate color
        var r = Math.round(colors.coldColor[0] +
          (colors.hotColor[0] - colors.coldColor[0]) * val);
        var g = Math.round(colors.coldColor[1] +
          (colors.hotColor[1] - colors.coldColor[1]) * val);
        var b = Math.round(colors.coldColor[2] +
          (colors.hotColor[2] - colors.coldColor[2]) * val);

        ctx.fillStyle = 'rgb(' + r + ',' + g + ',' + b + ')';
        ctx.fillRect(x, y, cellSize, cellSize);

        // Grid lines
        ctx.strokeStyle = colors.gridLine;
        ctx.lineWidth = 0.5;
        ctx.strokeRect(x, y, cellSize, cellSize);
      }
    }

    // Hover highlight
    if (hoverCell && hoverCell.row >= 0 && hoverCell.row < n &&
        hoverCell.col >= 0 && hoverCell.col < n) {
      var hx = labelMargin + hoverCell.col * cellSize;
      var hy = labelMargin + hoverCell.row * cellSize;

      // Row highlight
      ctx.fillStyle = 'rgba(255,255,255,0.08)';
      ctx.fillRect(labelMargin, hy, gridSize, cellSize);
      // Col highlight
      ctx.fillRect(hx, labelMargin, cellSize, gridSize);

      // Cell outline
      ctx.strokeStyle = colors.highlight;
      ctx.lineWidth = 2;
      ctx.strokeRect(hx, hy, cellSize, cellSize);
    }

    // Axis labels
    ctx.fillStyle = colors.textMuted;
    ctx.font = Math.min(11, cellSize * 0.6) + 'px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    for (var i = 0; i < n; i++) {
      // Column labels (top) — Key positions
      ctx.fillText(i, labelMargin + i * cellSize + cellSize / 2,
        labelMargin / 2);
      // Row labels (left) — Query positions
      ctx.fillText(i, labelMargin / 2,
        labelMargin + i * cellSize + cellSize / 2);
    }

    // Axis titles
    ctx.fillStyle = colors.text;
    ctx.font = 'bold 11px sans-serif';
    ctx.fillText('Key position (j)', labelMargin + gridSize / 2, 10);

    ctx.save();
    ctx.translate(10, labelMargin + gridSize / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Query position (i)', 0, 0);
    ctx.restore();

    return { cellSize: cellSize, labelMargin: labelMargin, gridSize: gridSize };
  }

  // --- Tooltip ---

  function showTooltip(container, x, y, row, col, value, colors) {
    var tip = container.querySelector('.attn-tooltip');
    if (!tip) {
      tip = document.createElement('div');
      tip.className = 'attn-tooltip';
      tip.style.cssText =
        'position:absolute;padding:6px 10px;border-radius:6px;' +
        'font:12px/1.4 monospace;pointer-events:none;z-index:100;' +
        'box-shadow:0 2px 8px rgba(0,0,0,0.15);transition:opacity 0.1s;';
      container.appendChild(tip);
    }
    tip.style.background = colors.tooltipBg;
    tip.style.border = '1px solid ' + colors.tooltipBorder;
    tip.style.color = colors.text;
    tip.innerHTML =
      'Query <b>' + row + '</b> → Key <b>' + col + '</b><br>' +
      'Weight: <b>' + value.toFixed(3) + '</b>';
    tip.style.opacity = '1';

    // Position near cursor but keep inside container
    var tipWidth = 160;
    var tipHeight = 48;
    var posX = x + 12;
    var posY = y - tipHeight - 8;
    if (posX + tipWidth > container.clientWidth) {
      posX = x - tipWidth - 12;
    }
    if (posY < 0) posY = y + 16;
    tip.style.left = posX + 'px';
    tip.style.top = posY + 'px';
  }

  function hideTooltip(container) {
    var tip = container.querySelector('.attn-tooltip');
    if (tip) tip.style.opacity = '0';
  }

  // --- Pattern descriptions ---

  var PATTERNS = {
    causal: {
      label: 'Causal',
      desc: 'Each token attends only to itself and previous tokens. ' +
        'Used in GPT, LLaMA, and all autoregressive models. ' +
        'The upper triangle is masked with -inf before softmax.',
      generate: function (n, params) { return causalPattern(n); }
    },
    sliding_window: {
      label: 'Sliding Window',
      desc: 'Each token attends to a fixed window of recent tokens. ' +
        'Used in Mistral/Mixtral to limit memory while keeping local context. ' +
        'Window size controls the attention span.',
      generate: function (n, params) {
        return slidingWindowPattern(n, params.windowSize);
      },
      params: [
        { key: 'windowSize', label: 'Window', min: 1, max: 16, default: 4 }
      ]
    },
    global_local: {
      label: 'Global + Local',
      desc: 'Combines a local sliding window with global tokens that attend ' +
        'everywhere. The first few tokens act as "summary" positions. ' +
        'Used in Longformer and BigBird architectures.',
      generate: function (n, params) {
        return globalLocalPattern(n, params.windowSize, params.globalTokens);
      },
      params: [
        { key: 'windowSize', label: 'Window', min: 1, max: 16, default: 4 },
        { key: 'globalTokens', label: 'Global', min: 1, max: 8, default: 2 }
      ]
    },
    strided: {
      label: 'Strided',
      desc: 'Attends to every N-th previous token. Combined with local ' +
        'attention in Sparse Transformers to achieve O(n*sqrt(n)) complexity. ' +
        'Stride=1 is equivalent to causal attention.',
      generate: function (n, params) {
        return stridedPattern(n, params.stride);
      },
      params: [
        { key: 'stride', label: 'Stride', min: 1, max: 8, default: 3 }
      ]
    },
    full: {
      label: 'Full (Bidirectional)',
      desc: 'Every token attends to every other token. ' +
        'Used in BERT-style encoder models. No causal masking — ' +
        'all positions see the full sequence.',
      generate: function (n, params) { return fullPattern(n); }
    }
  };

  // --- Widget Builder ---

  function createWidget(container) {
    var seqLen = parseInt(container.getAttribute('data-seq-len') || '12', 10);
    var currentPattern = 'causal';
    var paramValues = {};
    var showSoftmax = true;
    var hoverCell = null;
    var layoutInfo = null;

    container.style.position = 'relative';
    container.innerHTML = '';

    // Controls row
    var controls = document.createElement('div');
    controls.style.cssText =
      'display:flex;flex-wrap:wrap;gap:8px;align-items:center;' +
      'margin-bottom:12px;';

    // Pattern buttons
    var btnGroup = document.createElement('div');
    btnGroup.style.cssText = 'display:flex;flex-wrap:wrap;gap:4px;';

    Object.keys(PATTERNS).forEach(function (key) {
      var btn = document.createElement('button');
      btn.textContent = PATTERNS[key].label;
      btn.setAttribute('data-pattern', key);
      btn.style.cssText =
        'padding:4px 12px;border-radius:4px;border:1px solid;' +
        'cursor:pointer;font:13px sans-serif;transition:all 0.15s;';
      btn.addEventListener('click', function () {
        currentPattern = key;
        initParamDefaults();
        updateAll();
      });
      btnGroup.appendChild(btn);
    });
    controls.appendChild(btnGroup);
    container.appendChild(controls);

    // Seq length slider
    var seqControl = document.createElement('div');
    seqControl.style.cssText =
      'display:flex;align-items:center;gap:8px;margin-bottom:8px;';
    var seqLabel = document.createElement('label');
    seqLabel.style.cssText = 'font:13px sans-serif;white-space:nowrap;';
    seqLabel.textContent = 'Tokens: ' + seqLen;
    var seqSlider = document.createElement('input');
    seqSlider.type = 'range';
    seqSlider.min = '4';
    seqSlider.max = '24';
    seqSlider.value = String(seqLen);
    seqSlider.style.cssText = 'width:120px;';
    seqSlider.addEventListener('input', function () {
      seqLen = parseInt(this.value, 10);
      seqLabel.textContent = 'Tokens: ' + seqLen;
      updateAll();
    });
    seqControl.appendChild(seqLabel);
    seqControl.appendChild(seqSlider);

    // Softmax toggle
    var smToggle = document.createElement('label');
    smToggle.style.cssText =
      'font:13px sans-serif;display:flex;align-items:center;gap:4px;' +
      'margin-left:12px;cursor:pointer;';
    var smCheckbox = document.createElement('input');
    smCheckbox.type = 'checkbox';
    smCheckbox.checked = showSoftmax;
    smCheckbox.addEventListener('change', function () {
      showSoftmax = this.checked;
      updateAll();
    });
    smToggle.appendChild(smCheckbox);
    smToggle.appendChild(document.createTextNode('Apply softmax'));
    seqControl.appendChild(smToggle);

    container.appendChild(seqControl);

    // Dynamic param sliders container
    var paramRow = document.createElement('div');
    paramRow.style.cssText =
      'display:flex;flex-wrap:wrap;gap:12px;margin-bottom:8px;';
    container.appendChild(paramRow);

    // Description
    var descEl = document.createElement('p');
    descEl.style.cssText = 'font:13px/1.6 sans-serif;margin:0 0 12px 0;';
    container.appendChild(descEl);

    // Canvas
    var canvas = document.createElement('canvas');
    canvas.style.cssText = 'display:block;cursor:crosshair;';
    container.appendChild(canvas);

    // Mouse interaction
    canvas.addEventListener('mousemove', function (e) {
      if (!layoutInfo) return;
      var rect = canvas.getBoundingClientRect();
      var mx = e.clientX - rect.left;
      var my = e.clientY - rect.top;

      var col = Math.floor((mx - layoutInfo.labelMargin) / layoutInfo.cellSize);
      var row = Math.floor((my - layoutInfo.labelMargin) / layoutInfo.cellSize);

      if (row >= 0 && row < seqLen && col >= 0 && col < seqLen) {
        hoverCell = { row: row, col: col };
        var matrix = generateMatrix();
        var colors = getThemeColors();
        layoutInfo = renderHeatmap(canvas, matrix, colors, hoverCell);
        showTooltip(container, mx, my, row, col, matrix[row][col], colors);
      } else {
        hoverCell = null;
        var matrix = generateMatrix();
        layoutInfo = renderHeatmap(canvas, matrix, getThemeColors(), null);
        hideTooltip(container);
      }
    });

    canvas.addEventListener('mouseleave', function () {
      hoverCell = null;
      var matrix = generateMatrix();
      layoutInfo = renderHeatmap(canvas, matrix, getThemeColors(), null);
      hideTooltip(container);
    });

    function initParamDefaults() {
      var pat = PATTERNS[currentPattern];
      paramValues = {};
      if (pat.params) {
        pat.params.forEach(function (p) {
          paramValues[p.key] = p.default;
        });
      }
    }

    function generateMatrix() {
      var pat = PATTERNS[currentPattern];
      var mask = pat.generate(seqLen, paramValues);
      return showSoftmax ? applySoftmax(mask) : mask;
    }

    function updateAll() {
      var colors = getThemeColors();

      // Update button styles
      var buttons = btnGroup.querySelectorAll('button');
      buttons.forEach(function (btn) {
        var isActive = btn.getAttribute('data-pattern') === currentPattern;
        btn.style.background = isActive ? colors.controlActive : colors.controlBg;
        btn.style.color = isActive ? '#ffffff' : colors.text;
        btn.style.borderColor = isActive
          ? colors.controlActive : colors.controlBorder;
      });

      // Update description
      descEl.textContent = PATTERNS[currentPattern].desc;
      descEl.style.color = colors.textMuted;

      // Update param sliders
      paramRow.innerHTML = '';
      var pat = PATTERNS[currentPattern];
      if (pat.params) {
        pat.params.forEach(function (p) {
          var wrapper = document.createElement('div');
          wrapper.style.cssText =
            'display:flex;align-items:center;gap:6px;';
          var lab = document.createElement('label');
          lab.style.cssText = 'font:13px sans-serif;white-space:nowrap;';
          lab.style.color = colors.text;
          lab.textContent = p.label + ': ' + paramValues[p.key];
          var slider = document.createElement('input');
          slider.type = 'range';
          slider.min = String(p.min);
          slider.max = String(Math.min(p.max, seqLen));
          slider.value = String(paramValues[p.key]);
          slider.style.cssText = 'width:100px;';
          slider.addEventListener('input', function () {
            paramValues[p.key] = parseInt(this.value, 10);
            lab.textContent = p.label + ': ' + paramValues[p.key];
            updateAll();
          });
          wrapper.appendChild(lab);
          wrapper.appendChild(slider);
          paramRow.appendChild(wrapper);
        });
      }

      // Style other controls
      seqLabel.style.color = colors.text;
      smToggle.style.color = colors.text;

      // Render
      var matrix = generateMatrix();
      layoutInfo = renderHeatmap(canvas, matrix, colors, hoverCell);
    }

    initParamDefaults();
    updateAll();

    // Re-render on theme change
    var observer = new MutationObserver(function () { updateAll(); });
    observer.observe(document.body,
      { attributes: true, attributeFilter: ['data-md-color-scheme'] });

    // Re-render on resize
    window.addEventListener('resize', function () { updateAll(); });
  }

  // --- Init ---

  function init() {
    var containers = document.querySelectorAll('.attn-viz');
    containers.forEach(createWidget);
  }

  // Support mkdocs-material instant loading
  if (typeof document$ !== 'undefined') {
    document$.subscribe(function () { init(); });
  } else {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', init);
    } else {
      init();
    }
  }
})();
