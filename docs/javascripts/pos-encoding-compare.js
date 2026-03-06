/**
 * Positional Encoding Comparison Visualization
 *
 * Side-by-side comparison of different positional encoding strategies:
 *   - RoPE: Rotary Position Embedding (rotation-based, relative)
 *   - ALiBi: Attention with Linear Biases (slope-based, relative)
 *   - Sinusoidal: Classic absolute positional encoding
 *
 * Shows how each method creates position-dependent attention patterns.
 *
 * Built for LMT's educational documentation.
 */
;(function () {
  'use strict'

  // --- Theme ---
  function getThemeColors() {
    var isDark =
      document.body.getAttribute('data-md-color-scheme') === 'slate'
    return {
      bg: isDark ? '#1e1e2e' : '#fafafa',
      fg: isDark ? '#cdd6f4' : '#333333',
      accent: isDark ? '#cba6f7' : '#7c4dff',
      accent2: isDark ? '#f38ba8' : '#e91e63',
      accent3: isDark ? '#a6e3a1' : '#4caf50',
      accent4: isDark ? '#fab387' : '#ff9800',
      grid: isDark ? '#313244' : '#e0e0e0',
      dimmed: isDark ? '#6c7086' : '#9e9e9e',
      surface: isDark ? '#2a2a3e' : '#f0f0f0',
      heatLow: isDark ? '#1e1e2e' : '#ffffff',
      heatHigh: isDark ? '#cba6f7' : '#7c4dff',
    }
  }

  // --- Math ---
  function ropeFreq(dimIdx, halfD, base) {
    return 1.0 / Math.pow(base, dimIdx / halfD)
  }

  function ropeSimilarity(relDist, halfD, base) {
    var sum = 0
    for (var i = 0; i < halfD; i++) {
      sum += Math.cos(relDist * ropeFreq(i, halfD, base))
    }
    return sum / halfD
  }

  function alibiSlope(headIdx, numHeads) {
    var ratio = 8.0 / numHeads
    return Math.pow(2, -ratio * (headIdx + 1))
  }

  function alibiBias(relDist, headIdx, numHeads) {
    return -alibiSlope(headIdx, numHeads) * Math.abs(relDist)
  }

  function sinusoidalSimilarity(pos1, pos2, dModel) {
    // Dot product of sinusoidal encodings at two positions
    var sum = 0
    for (var i = 0; i < dModel; i++) {
      var div = Math.pow(10000, (2 * Math.floor(i / 2)) / dModel)
      var v1, v2
      if (i % 2 === 0) {
        v1 = Math.sin(pos1 / div)
        v2 = Math.sin(pos2 / div)
      } else {
        v1 = Math.cos(pos1 / div)
        v2 = Math.cos(pos2 / div)
      }
      sum += v1 * v2
    }
    return sum / dModel
  }

  // --- Canvas ---
  function createCanvas(container, width, height) {
    var canvas = document.createElement('canvas')
    var dpr = window.devicePixelRatio || 1
    canvas.width = width * dpr
    canvas.height = height * dpr
    canvas.style.width = width + 'px'
    canvas.style.height = height + 'px'
    var ctx = canvas.getContext('2d')
    ctx.scale(dpr, dpr)
    container.appendChild(canvas)
    return { canvas: canvas, ctx: ctx, width: width, height: height }
  }

  // --- Heatmap drawing ---
  function drawHeatmap(c, colors, title, matrix, seqLen, labels) {
    var ctx = c.ctx
    var w = c.width
    var h = c.height
    var margin = { top: 28, bottom: 24, left: 28, right: 12 }
    var plotW = w - margin.left - margin.right
    var plotH = h - margin.top - margin.bottom

    // Background
    ctx.fillStyle = colors.bg
    ctx.fillRect(0, 0, w, h)

    // Title
    ctx.fillStyle = colors.fg
    ctx.font = 'bold 12px monospace'
    ctx.textAlign = 'center'
    ctx.fillText(title, w / 2, 16)

    // Find min/max for normalization
    var minVal = Infinity
    var maxVal = -Infinity
    for (var i = 0; i < seqLen; i++) {
      for (var j = 0; j < seqLen; j++) {
        var v = matrix[i][j]
        if (v < minVal) minVal = v
        if (v > maxVal) maxVal = v
      }
    }
    var range = maxVal - minVal || 1

    // Draw cells
    var cellW = plotW / seqLen
    var cellH = plotH / seqLen
    for (var i = 0; i < seqLen; i++) {
      for (var j = 0; j < seqLen; j++) {
        var t = (matrix[i][j] - minVal) / range
        ctx.fillStyle = lerpColor(colors.heatLow, colors.heatHigh, t)
        ctx.fillRect(
          margin.left + j * cellW,
          margin.top + i * cellH,
          cellW + 0.5,
          cellH + 0.5
        )
      }
    }

    // Grid lines (sparse)
    ctx.strokeStyle = colors.grid
    ctx.lineWidth = 0.3
    var gridStep = Math.max(1, Math.floor(seqLen / 8))
    for (var i = 0; i <= seqLen; i += gridStep) {
      ctx.beginPath()
      ctx.moveTo(margin.left + i * cellW, margin.top)
      ctx.lineTo(margin.left + i * cellW, margin.top + plotH)
      ctx.stroke()
      ctx.beginPath()
      ctx.moveTo(margin.left, margin.top + i * cellH)
      ctx.lineTo(margin.left + plotW, margin.top + i * cellH)
      ctx.stroke()
    }

    // Labels
    if (labels) {
      ctx.fillStyle = colors.dimmed
      ctx.font = '9px monospace'
      ctx.textAlign = 'center'
      ctx.fillText(labels.x || '', w / 2, h - 4)
      ctx.save()
      ctx.translate(8, margin.top + plotH / 2)
      ctx.rotate(-Math.PI / 2)
      ctx.fillText(labels.y || '', 0, 0)
      ctx.restore()
    }
  }

  // --- Line chart for decay comparison ---
  function drawDecayChart(c, colors, data, seqLen, params) {
    var ctx = c.ctx
    var w = c.width
    var h = c.height
    var margin = { top: 28, bottom: 36, left: 50, right: 16 }
    var plotW = w - margin.left - margin.right
    var plotH = h - margin.top - margin.bottom

    // Background
    ctx.fillStyle = colors.bg
    ctx.fillRect(0, 0, w, h)

    // Title
    ctx.fillStyle = colors.fg
    ctx.font = 'bold 12px monospace'
    ctx.textAlign = 'center'
    ctx.fillText('Attention Bias vs. Relative Distance', w / 2, 16)

    // Plot background
    ctx.fillStyle = colors.surface
    ctx.fillRect(margin.left, margin.top, plotW, plotH)

    // Find global min/max
    var minVal = 0
    var maxVal = 0
    for (var key in data) {
      for (var i = 0; i < data[key].length; i++) {
        if (data[key][i] < minVal) minVal = data[key][i]
        if (data[key][i] > maxVal) maxVal = data[key][i]
      }
    }
    // Add some padding
    var yRange = (maxVal - minVal) || 1
    minVal -= yRange * 0.05
    maxVal += yRange * 0.05
    yRange = maxVal - minVal

    // Zero line
    var zeroY = margin.top + plotH * (1 - (0 - minVal) / yRange)
    ctx.beginPath()
    ctx.moveTo(margin.left, zeroY)
    ctx.lineTo(margin.left + plotW, zeroY)
    ctx.strokeStyle = colors.dimmed
    ctx.lineWidth = 0.8
    ctx.setLineDash([4, 3])
    ctx.stroke()
    ctx.setLineDash([])

    // Draw lines
    var lineColors = {
      rope: colors.accent,
      alibi: colors.accent2,
      sinusoidal: colors.accent3,
    }
    var lineLabels = {
      rope: 'RoPE',
      alibi: 'ALiBi (head 0)',
      sinusoidal: 'Sinusoidal',
    }

    var legendY = margin.top + 12
    for (var key in data) {
      var points = data[key]
      var numPoints = points.length

      ctx.beginPath()
      for (var i = 0; i < numPoints; i++) {
        var px = margin.left + (i / (numPoints - 1)) * plotW
        var py = margin.top + plotH * (1 - (points[i] - minVal) / yRange)
        if (i === 0) ctx.moveTo(px, py)
        else ctx.lineTo(px, py)
      }
      ctx.strokeStyle = lineColors[key]
      ctx.lineWidth = 2
      ctx.stroke()

      // Legend entry
      ctx.fillStyle = lineColors[key]
      ctx.fillRect(margin.left + 8, legendY - 4, 14, 3)
      ctx.fillStyle = colors.fg
      ctx.font = '10px monospace'
      ctx.textAlign = 'left'
      ctx.fillText(lineLabels[key], margin.left + 26, legendY)
      legendY += 14
    }

    // Axes
    ctx.fillStyle = colors.dimmed
    ctx.font = '10px monospace'
    ctx.textAlign = 'center'
    ctx.fillText('relative distance', w / 2, h - 4)

    ctx.save()
    ctx.translate(12, margin.top + plotH / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.textAlign = 'center'
    ctx.fillText('attention bias', 0, 0)
    ctx.restore()

    // Y ticks
    ctx.textAlign = 'right'
    for (var j = 0; j <= 4; j++) {
      var val = minVal + (yRange * j) / 4
      var yy = margin.top + plotH - (j / 4) * plotH
      ctx.fillStyle = colors.dimmed
      ctx.fillText(val.toFixed(2), margin.left - 4, yy + 3)
      ctx.beginPath()
      ctx.moveTo(margin.left, yy)
      ctx.lineTo(margin.left + plotW, yy)
      ctx.strokeStyle = colors.grid
      ctx.lineWidth = 0.3
      ctx.stroke()
    }

    // X ticks
    ctx.textAlign = 'center'
    for (var j = 0; j <= 4; j++) {
      var val = Math.round((seqLen * j) / 4)
      var xx = margin.left + (j / 4) * plotW
      ctx.fillText('' + val, xx, margin.top + plotH + 14)
    }
  }

  // --- ALiBi multi-head heatmap ---
  function drawAlibiHeads(c, colors, numHeads, seqLen) {
    var ctx = c.ctx
    var w = c.width
    var h = c.height
    var margin = { top: 28, bottom: 24, left: 8, right: 8 }

    ctx.fillStyle = colors.bg
    ctx.fillRect(0, 0, w, h)

    ctx.fillStyle = colors.fg
    ctx.font = 'bold 12px monospace'
    ctx.textAlign = 'center'
    ctx.fillText('ALiBi: Per-Head Slopes', w / 2, 16)

    // Draw a small heatmap for each head
    var cols = Math.min(numHeads, 4)
    var rows = Math.ceil(numHeads / cols)
    var availW = w - margin.left - margin.right
    var availH = h - margin.top - margin.bottom
    var gap = 6
    var cellAreaW = (availW - gap * (cols - 1)) / cols
    var cellAreaH = (availH - gap * (rows - 1)) / rows
    var cellSize = Math.min(cellAreaW, cellAreaH)

    for (var head = 0; head < numHeads; head++) {
      var col = head % cols
      var row = Math.floor(head / cols)
      var ox = margin.left + col * (cellSize + gap)
      var oy = margin.top + row * (cellSize + gap)

      // Compute bias matrix for this head
      var slope = alibiSlope(head, numHeads)
      var minBias = -slope * (seqLen - 1)

      var pixW = cellSize / seqLen
      var pixH = cellSize / seqLen

      for (var qi = 0; qi < seqLen; qi++) {
        for (var ki = 0; ki < seqLen; ki++) {
          var bias = -slope * Math.abs(qi - ki)
          var t = minBias === 0 ? 1 : (bias - minBias) / (0 - minBias)
          ctx.fillStyle = lerpColor(colors.heatLow, colors.heatHigh, t)
          ctx.fillRect(ox + ki * pixW, oy + qi * pixH, pixW + 0.5, pixH + 0.5)
        }
      }

      // Head label
      ctx.fillStyle = colors.dimmed
      ctx.font = '9px monospace'
      ctx.textAlign = 'center'
      ctx.fillText(
        'h' + head + ' (m=' + slope.toFixed(3) + ')',
        ox + cellSize / 2,
        oy + cellSize + 12
      )
    }
  }

  // --- Color helpers ---
  function lerpColor(c1, c2, t) {
    t = Math.max(0, Math.min(1, t))
    var r1 = parseInt(c1.slice(1, 3), 16)
    var g1 = parseInt(c1.slice(3, 5), 16)
    var b1 = parseInt(c1.slice(5, 7), 16)
    var r2 = parseInt(c2.slice(1, 3), 16)
    var g2 = parseInt(c2.slice(3, 5), 16)
    var b2 = parseInt(c2.slice(5, 7), 16)
    var r = Math.round(r1 + (r2 - r1) * t)
    var g = Math.round(g1 + (g2 - g1) * t)
    var b = Math.round(b1 + (b2 - b1) * t)
    return (
      '#' +
      r.toString(16).padStart(2, '0') +
      g.toString(16).padStart(2, '0') +
      b.toString(16).padStart(2, '0')
    )
  }

  // --- Slider ---
  function createSlider(container, label, min, max, value, step, onChange) {
    var wrapper = document.createElement('div')
    wrapper.style.cssText =
      'display:flex;align-items:center;gap:8px;margin:4px 0;font:12px monospace;'

    var lbl = document.createElement('label')
    lbl.textContent = label
    lbl.style.cssText = 'min-width:130px;color:inherit;'

    var input = document.createElement('input')
    input.type = 'range'
    input.min = min
    input.max = max
    input.value = value
    input.step = step
    input.style.cssText = 'flex:1;'

    var display = document.createElement('span')
    display.textContent = value
    display.style.cssText = 'min-width:48px;text-align:right;color:inherit;'

    input.addEventListener('input', function () {
      display.textContent = input.value
      onChange(Number(input.value))
    })

    wrapper.appendChild(lbl)
    wrapper.appendChild(input)
    wrapper.appendChild(display)
    container.appendChild(wrapper)
    return input
  }

  // --- Main ---
  function initCompareViz(container) {
    var colors = getThemeColors()

    var params = {
      seqLen: parseInt(container.getAttribute('data-seq-len') || '32'),
      dModel: parseInt(container.getAttribute('data-d-model') || '64'),
      numHeads: parseInt(container.getAttribute('data-num-heads') || '8'),
      ropeBase: 10000,
    }

    container.style.cssText =
      'padding:16px;border-radius:8px;background:' +
      colors.bg +
      ';color:' +
      colors.fg +
      ';'

    // Controls
    var controls = document.createElement('div')
    controls.style.cssText = 'margin-bottom:12px;'
    container.appendChild(controls)

    createSlider(
      controls,
      'Sequence length:',
      8,
      64,
      params.seqLen,
      1,
      function (v) {
        params.seqLen = v
        redraw()
      }
    )

    createSlider(
      controls,
      'Num heads (ALiBi):',
      1,
      16,
      params.numHeads,
      1,
      function (v) {
        params.numHeads = v
        redraw()
      }
    )

    createSlider(
      controls,
      'RoPE base:',
      100,
      100000,
      params.ropeBase,
      100,
      function (v) {
        params.ropeBase = v
        redraw()
      }
    )

    // Row 1: Heatmaps
    var heatRow = document.createElement('div')
    heatRow.style.cssText =
      'display:flex;flex-wrap:wrap;gap:12px;justify-content:center;margin-bottom:12px;'
    container.appendChild(heatRow)

    var heatSize = 240
    var ropeHeatDiv = document.createElement('div')
    heatRow.appendChild(ropeHeatDiv)
    var ropeHeatCanvas = createCanvas(ropeHeatDiv, heatSize, heatSize)

    var alibiHeatDiv = document.createElement('div')
    heatRow.appendChild(alibiHeatDiv)
    var alibiHeatCanvas = createCanvas(alibiHeatDiv, heatSize, heatSize)

    var sinHeatDiv = document.createElement('div')
    heatRow.appendChild(sinHeatDiv)
    var sinHeatCanvas = createCanvas(sinHeatDiv, heatSize, heatSize)

    // Row 2: Decay comparison + ALiBi heads
    var row2 = document.createElement('div')
    row2.style.cssText =
      'display:flex;flex-wrap:wrap;gap:12px;justify-content:center;'
    container.appendChild(row2)

    var decayDiv = document.createElement('div')
    row2.appendChild(decayDiv)
    var decayCanvas = createCanvas(decayDiv, 380, 240)

    var alibiHeadsDiv = document.createElement('div')
    row2.appendChild(alibiHeadsDiv)
    var alibiHeadsCanvas = createCanvas(alibiHeadsDiv, 340, 240)

    // Comparison table
    var tableDiv = document.createElement('div')
    tableDiv.style.cssText =
      'margin-top:12px;padding:10px;border-radius:4px;background:' +
      colors.surface +
      ';font:11px monospace;line-height:1.6;overflow-x:auto;'
    tableDiv.innerHTML =
      '<table style="width:100%;border-collapse:collapse;">' +
      '<tr style="border-bottom:1px solid ' +
      colors.grid +
      '">' +
      '<th style="text-align:left;padding:4px 8px;">Property</th>' +
      '<th style="text-align:center;padding:4px 8px;">RoPE</th>' +
      '<th style="text-align:center;padding:4px 8px;">ALiBi</th>' +
      '<th style="text-align:center;padding:4px 8px;">Sinusoidal</th></tr>' +
      '<tr><td style="padding:4px 8px">Position type</td>' +
      '<td style="text-align:center">Relative</td>' +
      '<td style="text-align:center">Relative</td>' +
      '<td style="text-align:center">Absolute</td></tr>' +
      '<tr><td style="padding:4px 8px">Mechanism</td>' +
      '<td style="text-align:center">Rotation</td>' +
      '<td style="text-align:center">Linear bias</td>' +
      '<td style="text-align:center">Addition</td></tr>' +
      '<tr><td style="padding:4px 8px">Learnable params</td>' +
      '<td style="text-align:center">0</td>' +
      '<td style="text-align:center">0</td>' +
      '<td style="text-align:center">0 or d_model</td></tr>' +
      '<tr><td style="padding:4px 8px">Applied to</td>' +
      '<td style="text-align:center">Q, K</td>' +
      '<td style="text-align:center">Attention scores</td>' +
      '<td style="text-align:center">Input</td></tr>' +
      '<tr><td style="padding:4px 8px">Length extrapolation</td>' +
      '<td style="text-align:center">Good (with scaling)</td>' +
      '<td style="text-align:center">Excellent</td>' +
      '<td style="text-align:center">Poor</td></tr>' +
      '<tr><td style="padding:4px 8px">Used in</td>' +
      '<td style="text-align:center">LLaMA, Mixtral, MLA</td>' +
      '<td style="text-align:center">BLOOM, MPT</td>' +
      '<td style="text-align:center">Original Transformer</td></tr>' +
      '</table>'
    container.appendChild(tableDiv)

    function computeMatrices() {
      var sl = params.seqLen
      var halfD = params.dModel / 2

      // RoPE similarity matrix
      var ropeMat = []
      for (var i = 0; i < sl; i++) {
        ropeMat[i] = []
        for (var j = 0; j < sl; j++) {
          ropeMat[i][j] = ropeSimilarity(
            Math.abs(i - j),
            halfD,
            params.ropeBase
          )
        }
      }

      // ALiBi bias matrix (head 0 -- strongest slope)
      var alibiMat = []
      for (var i = 0; i < sl; i++) {
        alibiMat[i] = []
        for (var j = 0; j < sl; j++) {
          alibiMat[i][j] = alibiBias(i - j, 0, params.numHeads)
        }
      }

      // Sinusoidal similarity matrix
      var sinMat = []
      for (var i = 0; i < sl; i++) {
        sinMat[i] = []
        for (var j = 0; j < sl; j++) {
          sinMat[i][j] = sinusoidalSimilarity(i, j, params.dModel)
        }
      }

      return { rope: ropeMat, alibi: alibiMat, sinusoidal: sinMat }
    }

    function computeDecayData() {
      var sl = params.seqLen
      var halfD = params.dModel / 2
      var ropeDecay = []
      var alibiDecay = []
      var sinDecay = []

      for (var d = 0; d < sl; d++) {
        ropeDecay.push(ropeSimilarity(d, halfD, params.ropeBase))
        alibiDecay.push(alibiBias(d, 0, params.numHeads))
        sinDecay.push(sinusoidalSimilarity(0, d, params.dModel))
      }

      return { rope: ropeDecay, alibi: alibiDecay, sinusoidal: sinDecay }
    }

    function redraw() {
      colors = getThemeColors()
      container.style.background = colors.bg
      container.style.color = colors.fg
      tableDiv.style.background = colors.surface

      var matrices = computeMatrices()
      drawHeatmap(ropeHeatCanvas, colors, 'RoPE Similarity', matrices.rope, params.seqLen, {
        x: 'key position',
        y: 'query position',
      })
      drawHeatmap(alibiHeatCanvas, colors, 'ALiBi Bias (head 0)', matrices.alibi, params.seqLen, {
        x: 'key position',
        y: 'query position',
      })
      drawHeatmap(
        sinHeatCanvas,
        colors,
        'Sinusoidal Similarity',
        matrices.sinusoidal,
        params.seqLen,
        { x: 'key position', y: 'query position' }
      )

      var decayData = computeDecayData()
      drawDecayChart(decayCanvas, colors, decayData, params.seqLen, params)
      drawAlibiHeads(alibiHeadsCanvas, colors, params.numHeads, params.seqLen)
    }

    redraw()

    // Theme observer
    var observer = new MutationObserver(function () {
      redraw()
    })
    observer.observe(document.body, {
      attributes: true,
      attributeFilter: ['data-md-color-scheme'],
    })
  }

  // --- Bootstrap ---
  function boot() {
    var containers = document.querySelectorAll('.pos-encoding-compare')
    containers.forEach(initCompareViz)
  }

  if (typeof document$ !== 'undefined') {
    document$.subscribe(boot)
  } else {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', boot)
    } else {
      boot()
    }
  }
})()
