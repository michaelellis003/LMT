/**
 * RoPE (Rotary Position Embedding) Interactive Visualization
 *
 * Shows how RoPE encodes position through rotation of dimension pairs.
 * Three panels:
 *   1. Rotation circles -- how a single dimension pair rotates with position
 *   2. Frequency spectrum -- different dimension pairs rotate at different speeds
 *   3. Dot-product decay -- relative position encoding property
 *
 * Built for LMT's educational documentation.
 */
;(function () {
  'use strict'

  // --- Theme detection ---
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
    }
  }

  // --- Math helpers ---
  function ropeFreq(dim_idx, half_d, base) {
    return 1.0 / Math.pow(base, dim_idx / half_d)
  }

  // --- Canvas utilities ---
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

  function clearCanvas(c, colors) {
    c.ctx.fillStyle = colors.bg
    c.ctx.fillRect(0, 0, c.width, c.height)
  }

  // --- Panel 1: Rotation Circles ---
  function drawRotationPanel(c, colors, position, dimPair, params) {
    var ctx = c.ctx
    var w = c.width
    var h = c.height

    clearCanvas(c, colors)

    var centerX = w / 2
    var centerY = h / 2
    var radius = Math.min(w, h) * 0.32

    // Draw circle
    ctx.beginPath()
    ctx.arc(centerX, centerY, radius, 0, Math.PI * 2)
    ctx.strokeStyle = colors.grid
    ctx.lineWidth = 1.5
    ctx.stroke()

    // Draw axes
    ctx.beginPath()
    ctx.moveTo(centerX - radius - 10, centerY)
    ctx.lineTo(centerX + radius + 10, centerY)
    ctx.moveTo(centerX, centerY - radius - 10)
    ctx.lineTo(centerX, centerY + radius + 10)
    ctx.strokeStyle = colors.dimmed
    ctx.lineWidth = 0.8
    ctx.stroke()

    // Axis labels
    ctx.fillStyle = colors.dimmed
    ctx.font = '11px monospace'
    ctx.textAlign = 'center'
    ctx.fillText('x\u2081', centerX + radius + 18, centerY + 4)
    ctx.fillText('x\u2082', centerX, centerY - radius - 14)

    var half_d = params.d_model / 2
    var freq = ropeFreq(dimPair, half_d, params.base)

    // Draw multiple positions as faded dots to show the trajectory
    var maxShow = Math.min(position + 1, params.max_pos)
    for (var p = 0; p < maxShow; p++) {
      var angle = p * freq
      var px = centerX + radius * Math.cos(angle)
      var py = centerY - radius * Math.sin(angle)
      var alpha = p === position ? 1.0 : 0.15 + 0.15 * (p / maxShow)
      var size = p === position ? 6 : 3

      ctx.beginPath()
      ctx.arc(px, py, size, 0, Math.PI * 2)
      ctx.fillStyle =
        p === position
          ? colors.accent
          : colors.accent + hexAlpha(alpha)
      ctx.fill()

      // Label current position
      if (p === position) {
        ctx.beginPath()
        ctx.moveTo(centerX, centerY)
        ctx.lineTo(px, py)
        ctx.strokeStyle = colors.accent
        ctx.lineWidth = 1.5
        ctx.setLineDash([4, 3])
        ctx.stroke()
        ctx.setLineDash([])

        // Angle arc
        if (Math.abs(angle) > 0.05) {
          var arcR = radius * 0.25
          ctx.beginPath()
          ctx.arc(centerX, centerY, arcR, 0, -angle, angle > 0)
          ctx.strokeStyle = colors.accent2
          ctx.lineWidth = 1.5
          ctx.stroke()
        }
      }
    }

    // Title
    ctx.fillStyle = colors.fg
    ctx.font = 'bold 13px monospace'
    ctx.textAlign = 'center'
    ctx.fillText('Rotation for dim pair ' + dimPair, centerX, 20)

    // Info
    ctx.font = '11px monospace'
    ctx.fillStyle = colors.dimmed
    ctx.textAlign = 'left'
    ctx.fillText(
      '\u03B8 = ' + freq.toFixed(4) + ' rad/pos',
      8,
      h - 28
    )
    ctx.fillText(
      'pos=' + position + '  angle=' + (position * freq).toFixed(2) + ' rad',
      8,
      h - 12
    )
  }

  // --- Panel 2: Frequency Spectrum ---
  function drawFrequencyPanel(c, colors, position, params) {
    var ctx = c.ctx
    var w = c.width
    var h = c.height

    clearCanvas(c, colors)

    var half_d = params.d_model / 2
    var margin = { top: 32, bottom: 36, left: 50, right: 16 }
    var plotW = w - margin.left - margin.right
    var plotH = h - margin.top - margin.bottom

    // Title
    ctx.fillStyle = colors.fg
    ctx.font = 'bold 13px monospace'
    ctx.textAlign = 'center'
    ctx.fillText('Frequency Spectrum (\u03B8\u1D62 per dim pair)', w / 2, 18)

    // Background
    ctx.fillStyle = colors.surface
    ctx.fillRect(margin.left, margin.top, plotW, plotH)

    // Compute frequencies
    var freqs = []
    var maxFreq = 0
    for (var i = 0; i < half_d; i++) {
      var f = ropeFreq(i, half_d, params.base)
      freqs.push(f)
      if (f > maxFreq) maxFreq = f
    }

    // Draw bars
    var barW = Math.max(2, (plotW - half_d) / half_d)
    var gap = (plotW - barW * half_d) / (half_d + 1)

    for (var i = 0; i < half_d; i++) {
      var barH = (freqs[i] / maxFreq) * (plotH - 8)
      var bx = margin.left + gap + i * (barW + gap)
      var by = margin.top + plotH - barH

      // Color: gradient from accent (low freq) to accent2 (high freq)
      var t = i / (half_d - 1)
      ctx.fillStyle = lerpColor(colors.accent3, colors.accent2, t)
      ctx.fillRect(bx, by, barW, barH)

      // Rotation indicator: small arc showing how much this dim has rotated
      var totalAngle = position * freqs[i]
      var wraps = totalAngle / (Math.PI * 2)
      if (wraps > 0.05) {
        var indicatorR = Math.min(barW * 0.8, 8)
        var ix = bx + barW / 2
        var iy = by - indicatorR - 3
        if (iy > margin.top + indicatorR + 2) {
          ctx.beginPath()
          ctx.arc(
            ix,
            iy,
            indicatorR,
            0,
            -Math.min(totalAngle % (Math.PI * 2), Math.PI * 2),
            true
          )
          ctx.strokeStyle = colors.accent4
          ctx.lineWidth = 1.5
          ctx.stroke()
        }
      }
    }

    // X axis
    ctx.fillStyle = colors.dimmed
    ctx.font = '10px monospace'
    ctx.textAlign = 'center'
    ctx.fillText('dimension pair index', w / 2, h - 4)

    // Y axis
    ctx.save()
    ctx.translate(12, margin.top + plotH / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.textAlign = 'center'
    ctx.fillText('frequency \u03B8\u1D62', 0, 0)
    ctx.restore()

    // Y ticks
    ctx.textAlign = 'right'
    for (var j = 0; j <= 4; j++) {
      var val = (maxFreq * j) / 4
      var yy = margin.top + plotH - (j / 4) * (plotH - 8)
      ctx.fillStyle = colors.dimmed
      ctx.fillText(val.toFixed(3), margin.left - 4, yy + 3)
      ctx.beginPath()
      ctx.moveTo(margin.left, yy)
      ctx.lineTo(margin.left + plotW, yy)
      ctx.strokeStyle = colors.grid
      ctx.lineWidth = 0.5
      ctx.stroke()
    }

    // Legend for rotation indicators
    ctx.fillStyle = colors.dimmed
    ctx.font = '10px monospace'
    ctx.textAlign = 'left'
    ctx.fillText(
      'Arcs show cumulative rotation at pos=' + position,
      margin.left,
      h - 4
    )
  }

  // --- Panel 3: Dot-Product Decay ---
  function drawDotProductPanel(c, colors, params) {
    var ctx = c.ctx
    var w = c.width
    var h = c.height

    clearCanvas(c, colors)

    var half_d = params.d_model / 2
    var margin = { top: 32, bottom: 36, left: 50, right: 16 }
    var plotW = w - margin.left - margin.right
    var plotH = h - margin.top - margin.bottom

    // Title
    ctx.fillStyle = colors.fg
    ctx.font = 'bold 13px monospace'
    ctx.textAlign = 'center'
    ctx.fillText(
      'Relative Position: \u27E8R(m)q, R(n)k\u27E9 \u221D cos(\u0394pos\u00B7\u03B8)',
      w / 2,
      18
    )

    // Background
    ctx.fillStyle = colors.surface
    ctx.fillRect(margin.left, margin.top, plotW, plotH)

    // Compute average cos similarity across all dim pairs for each relative distance
    var maxDist = params.max_pos
    var numPoints = Math.min(maxDist, 200)
    var points = []
    var minVal = 1
    var maxVal = -1

    for (var d = 0; d < numPoints; d++) {
      var sum = 0
      for (var i = 0; i < half_d; i++) {
        var freq = ropeFreq(i, half_d, params.base)
        sum += Math.cos(d * freq)
      }
      var avg = sum / half_d
      points.push(avg)
      if (avg < minVal) minVal = avg
      if (avg > maxVal) maxVal = avg
    }

    // Ensure y range includes 0
    if (minVal > 0) minVal = -0.1
    if (maxVal < 0) maxVal = 0.1
    var yRange = maxVal - minVal

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

    // Draw the curve
    ctx.beginPath()
    for (var d = 0; d < numPoints; d++) {
      var px = margin.left + (d / (numPoints - 1)) * plotW
      var py = margin.top + plotH * (1 - (points[d] - minVal) / yRange)
      if (d === 0) ctx.moveTo(px, py)
      else ctx.lineTo(px, py)
    }
    ctx.strokeStyle = colors.accent
    ctx.lineWidth = 2
    ctx.stroke()

    // Fill under curve to zero
    ctx.lineTo(margin.left + plotW, zeroY)
    ctx.lineTo(margin.left, zeroY)
    ctx.closePath()
    ctx.fillStyle = colors.accent + '22'
    ctx.fill()

    // X axis label
    ctx.fillStyle = colors.dimmed
    ctx.font = '10px monospace'
    ctx.textAlign = 'center'
    ctx.fillText('relative distance |m - n|', w / 2, h - 4)

    // Y axis label
    ctx.save()
    ctx.translate(12, margin.top + plotH / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.textAlign = 'center'
    ctx.fillText('avg cos similarity', 0, 0)
    ctx.restore()

    // Y ticks
    ctx.textAlign = 'right'
    for (var j = 0; j <= 4; j++) {
      var val = minVal + (yRange * j) / 4
      var yy = margin.top + plotH - (j / 4) * plotH
      ctx.fillStyle = colors.dimmed
      ctx.font = '10px monospace'
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
      var val = Math.round((numPoints * j) / 4)
      var xx = margin.left + (j / 4) * plotW
      ctx.fillText('' + val, xx, margin.top + plotH + 14)
    }

    // Annotation: the key insight
    ctx.fillStyle = colors.accent3
    ctx.font = '11px monospace'
    ctx.textAlign = 'left'
    ctx.fillText(
      'Peak at \u0394=0: nearby positions are most similar',
      margin.left + 4,
      margin.top + 14
    )
  }

  // --- Color helpers ---
  function hexAlpha(a) {
    var hex = Math.round(a * 255)
      .toString(16)
      .padStart(2, '0')
    return hex
  }

  function lerpColor(c1, c2, t) {
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

  // --- Slider creation ---
  function createSlider(container, label, min, max, value, step, onChange) {
    var wrapper = document.createElement('div')
    wrapper.style.cssText =
      'display:flex;align-items:center;gap:8px;margin:4px 0;font:12px monospace;'

    var lbl = document.createElement('label')
    lbl.textContent = label
    lbl.style.cssText = 'min-width:120px;color:inherit;'

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

  // --- Main initialization ---
  function initRoPEViz(container) {
    var colors = getThemeColors()

    // Parameters
    var params = {
      d_model: parseInt(container.getAttribute('data-d-model') || '64'),
      base: parseFloat(container.getAttribute('data-base') || '10000'),
      max_pos: parseInt(container.getAttribute('data-max-pos') || '128'),
    }
    var position = 0
    var dimPair = 0

    // Layout
    container.style.cssText =
      'padding:16px;border-radius:8px;background:' +
      colors.bg +
      ';color:' +
      colors.fg +
      ';'

    // Controls area
    var controls = document.createElement('div')
    controls.style.cssText = 'margin-bottom:12px;'
    container.appendChild(controls)

    var posSlider = createSlider(
      controls,
      'Position (m):',
      0,
      params.max_pos - 1,
      0,
      1,
      function (v) {
        position = v
        redraw()
      }
    )

    var dimSlider = createSlider(
      controls,
      'Dim pair (i):',
      0,
      params.d_model / 2 - 1,
      0,
      1,
      function (v) {
        dimPair = v
        redraw()
      }
    )

    var baseSlider = createSlider(
      controls,
      'Base (\u03B8 base):',
      100,
      100000,
      params.base,
      100,
      function (v) {
        params.base = v
        redraw()
      }
    )

    // Canvas panels
    var panelRow = document.createElement('div')
    panelRow.style.cssText =
      'display:flex;flex-wrap:wrap;gap:12px;justify-content:center;'
    container.appendChild(panelRow)

    // Panel sizing
    var panelSize = 280

    var rotDiv = document.createElement('div')
    panelRow.appendChild(rotDiv)
    var rotCanvas = createCanvas(rotDiv, panelSize, panelSize)

    var freqDiv = document.createElement('div')
    panelRow.appendChild(freqDiv)
    var freqCanvas = createCanvas(freqDiv, panelSize + 60, panelSize)

    var dotDiv = document.createElement('div')
    panelRow.appendChild(dotDiv)
    var dotCanvas = createCanvas(dotDiv, panelSize + 60, panelSize)

    // Formula display
    var formulaDiv = document.createElement('div')
    formulaDiv.style.cssText =
      'margin-top:12px;padding:10px;border-radius:4px;background:' +
      colors.surface +
      ';font:12px monospace;line-height:1.6;'
    formulaDiv.innerHTML =
      '<b>RoPE formula:</b> ' +
      '\u03B8\u1D62 = base<sup>-2i/d</sup> &nbsp;&nbsp;|&nbsp;&nbsp; ' +
      'R(\u03B8,m)\u00B7x = [x\u2081 cos(m\u03B8) - x\u2082 sin(m\u03B8), ' +
      'x\u2082 cos(m\u03B8) + x\u2081 sin(m\u03B8)]<br>' +
      '<b>Key property:</b> ' +
      '\u27E8R(m)q, R(n)k\u27E9 = \u27E8R(m-n)q, k\u27E9 ' +
      '&mdash; dot product depends only on <em>relative</em> position'
    container.appendChild(formulaDiv)

    function redraw() {
      colors = getThemeColors()
      container.style.background = colors.bg
      container.style.color = colors.fg
      formulaDiv.style.background = colors.surface

      drawRotationPanel(rotCanvas, colors, position, dimPair, params)
      drawFrequencyPanel(freqCanvas, colors, position, params)
      drawDotProductPanel(dotCanvas, colors, params)
    }

    redraw()

    // Theme change observer
    var observer = new MutationObserver(function () {
      redraw()
    })
    observer.observe(document.body, {
      attributes: true,
      attributeFilter: ['data-md-color-scheme'],
    })

    // Animation: auto-advance position
    var animBtn = document.createElement('button')
    animBtn.textContent = '\u25B6 Animate'
    animBtn.style.cssText =
      'margin-top:8px;padding:6px 16px;border:1px solid ' +
      colors.dimmed +
      ';border-radius:4px;background:' +
      colors.surface +
      ';color:' +
      colors.fg +
      ';font:12px monospace;cursor:pointer;'
    container.appendChild(animBtn)

    var animId = null
    animBtn.addEventListener('click', function () {
      if (animId) {
        cancelAnimationFrame(animId)
        animId = null
        animBtn.textContent = '\u25B6 Animate'
        return
      }
      animBtn.textContent = '\u23F8 Pause'
      var lastTime = 0
      function step(ts) {
        if (ts - lastTime > 100) {
          lastTime = ts
          position = (position + 1) % params.max_pos
          posSlider.value = position
          posSlider.dispatchEvent(new Event('input'))
        }
        animId = requestAnimationFrame(step)
      }
      animId = requestAnimationFrame(step)
    })
  }

  // --- Bootstrap ---
  function boot() {
    var containers = document.querySelectorAll('.rope-viz')
    containers.forEach(initRoPEViz)
  }

  // Support mkdocs-material instant loading
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
