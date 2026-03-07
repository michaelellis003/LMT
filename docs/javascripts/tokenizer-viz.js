/**
 * LMT Tokenizer Comparison Visualization
 *
 * Interactive visualization showing how different tokenization strategies
 * split the same text. Highlights the trade-off between vocabulary size
 * and sequence length.
 *
 * Usage: Add a <div class="tokenizer-viz"></div> to any markdown page.
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
      controlBg: dark ? '#313244' : '#f5f5f5',
      controlBorder: dark ? '#45475a' : '#cccccc',
      controlActive: dark ? '#cba6f7' : '#6a1b9a',
      // Token highlight colors (repeating palette)
      tokenColors: dark
        ? [
            '#f38ba8', '#a6e3a1', '#89b4fa', '#f9e2af',
            '#cba6f7', '#94e2d5', '#fab387', '#89dceb',
            '#f5c2e7', '#74c7ec', '#eba0ac', '#a6adc8',
          ]
        : [
            '#e74c3c', '#27ae60', '#2980b9', '#f39c12',
            '#8e44ad', '#16a085', '#d35400', '#2c3e50',
            '#c0392b', '#1abc9c', '#e67e22', '#7f8c8d',
          ],
    };
  }

  // --- Simulated Tokenizers ---

  // Character-level: each character is a token
  function charTokenize(text) {
    return text.split('').map(function (ch) { return ch; });
  }

  // Word-level: split on whitespace and punctuation
  function wordTokenize(text) {
    var tokens = [];
    var current = '';
    for (var i = 0; i < text.length; i++) {
      var ch = text[i];
      if (/\s/.test(ch)) {
        if (current) tokens.push(current);
        tokens.push(ch);
        current = '';
      } else if (/[.,!?;:'"()\-]/.test(ch)) {
        if (current) tokens.push(current);
        tokens.push(ch);
        current = '';
      } else {
        current += ch;
      }
    }
    if (current) tokens.push(current);
    return tokens;
  }

  // BPE-like: simulate subword tokenization with common English subwords
  function bpeTokenize(text) {
    // Common BPE merges (simplified simulation)
    var merges = [
      'the', 'ing', 'tion', 'ed', 'er', 'es', 'al', 'en',
      'an', 'on', 'at', 'or', 'is', 'it', 'in', 'of',
      'and', 'for', 'was', 'are', 'not', 'you', 'all',
      'her', 'had', 'one', 'our', 'out', 'day', 'get',
      'has', 'him', 'his', 'how', 'its', 'may', 'new',
      'now', 'old', 'see', 'way', 'who', 'did', 'let',
      'say', 'she', 'too', 'use', 'with', 'have', 'from',
      'that', 'they', 'been', 'said', 'each', 'time',
      'very', 'when', 'come', 'make', 'like', 'long',
      'look', 'many', 'some', 'them', 'than', 'first',
      'could', 'would', 'people', 'little', 'once', 'upon',
    ];
    // Sort by length descending for greedy matching
    merges.sort(function (a, b) { return b.length - a.length; });

    var tokens = [];
    var i = 0;
    while (i < text.length) {
      // Handle whitespace
      if (/\s/.test(text[i])) {
        tokens.push(text[i]);
        i++;
        continue;
      }
      // Handle punctuation
      if (/[.,!?;:'"()\-]/.test(text[i])) {
        tokens.push(text[i]);
        i++;
        continue;
      }
      // Try to match longest BPE token
      var matched = false;
      var remaining = text.slice(i).toLowerCase();
      for (var m = 0; m < merges.length; m++) {
        var merge = merges[m];
        if (remaining.indexOf(merge) === 0) {
          tokens.push(text.slice(i, i + merge.length));
          i += merge.length;
          matched = true;
          break;
        }
      }
      if (!matched) {
        // Fall back to single character
        tokens.push(text[i]);
        i++;
      }
    }
    return tokens;
  }

  var TOKENIZERS = {
    'Character': { fn: charTokenize, desc: 'Each character is one token. Tiny vocabulary (~100), very long sequences.' },
    'Word': { fn: wordTokenize, desc: 'Split on spaces and punctuation. Large vocabulary, short sequences.' },
    'BPE (subword)': { fn: bpeTokenize, desc: 'Merge frequent character pairs. Balanced vocabulary and sequence length.' },
  };

  var EXAMPLE_TEXTS = [
    'Once upon a time, there was a little cat.',
    'The transformer architecture uses multi-head attention.',
    'She looked at the beautiful flowers in the garden.',
    'Programming language models is fascinating work!',
  ];

  // --- Rendering ---

  function renderTokens(container, tokens, colors, label) {
    var row = document.createElement('div');
    row.style.cssText = 'margin-bottom: 12px;';

    var header = document.createElement('div');
    header.style.cssText = 'font-size: 13px; color: ' + colors.textMuted +
      '; margin-bottom: 4px; font-family: monospace;';
    header.textContent = label + ' (' + tokens.length + ' tokens)';
    row.appendChild(header);

    var tokenRow = document.createElement('div');
    tokenRow.style.cssText = 'display: flex; flex-wrap: wrap; gap: 2px; align-items: center;';

    var colorIdx = 0;
    for (var i = 0; i < tokens.length; i++) {
      var token = tokens[i];
      var span = document.createElement('span');

      // Skip coloring for pure whitespace
      if (/^\s+$/.test(token)) {
        span.style.cssText = 'display: inline-block; width: ' +
          (token.length * 6) + 'px; height: 28px;';
      } else {
        var bgColor = colors.tokenColors[colorIdx % colors.tokenColors.length];
        colorIdx++;
        span.style.cssText =
          'display: inline-block; padding: 2px 4px; border-radius: 3px; ' +
          'font-family: monospace; font-size: 14px; line-height: 24px; ' +
          'background: ' + bgColor + '20; border: 1px solid ' + bgColor + '60; ' +
          'color: ' + colors.text + '; cursor: default; white-space: pre;';
        // Show special chars visually
        var display = token
          .replace(/ /g, '\u00B7')
          .replace(/\n/g, '\u21B5')
          .replace(/\t/g, '\u2192');
        span.textContent = display;
        span.title = 'Token ' + i + ': "' + token + '" (' + token.length + ' chars)';
      }
      tokenRow.appendChild(span);
    }
    row.appendChild(tokenRow);
    container.appendChild(row);
  }

  function renderStats(container, results, colors) {
    var table = document.createElement('table');
    table.style.cssText = 'width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 14px;';

    var thead = document.createElement('thead');
    var headerRow = document.createElement('tr');
    ['Tokenizer', 'Tokens', 'Avg Token Length', 'Compression'].forEach(function (h) {
      var th = document.createElement('th');
      th.style.cssText = 'text-align: left; padding: 6px 8px; border-bottom: 2px solid ' +
        colors.controlBorder + '; color: ' + colors.text + ';';
      th.textContent = h;
      headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);

    var tbody = document.createElement('tbody');
    var charCount = results[0].tokens.length; // Character tokenizer is baseline

    results.forEach(function (r) {
      var tr = document.createElement('tr');
      var avgLen = 0;
      var nonSpace = r.tokens.filter(function (t) { return !/^\s+$/.test(t); });
      if (nonSpace.length > 0) {
        var totalChars = nonSpace.reduce(function (sum, t) { return sum + t.length; }, 0);
        avgLen = totalChars / nonSpace.length;
      }
      var compression = (charCount / r.tokens.length).toFixed(1);

      [r.name, r.tokens.length, avgLen.toFixed(1) + ' chars', compression + 'x'].forEach(function (val) {
        var td = document.createElement('td');
        td.style.cssText = 'padding: 6px 8px; border-bottom: 1px solid ' +
          colors.controlBorder + '; color: ' + colors.text + ';';
        td.textContent = val;
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    container.appendChild(table);
  }

  // --- Main Setup ---

  function initViz(root) {
    var colors = getThemeColors();

    root.style.cssText = 'padding: 20px; background: ' + colors.bg +
      '; border-radius: 8px; border: 1px solid ' + colors.controlBorder + ';';
    root.innerHTML = '';

    // Title
    var title = document.createElement('h3');
    title.style.cssText = 'margin: 0 0 16px 0; color: ' + colors.text + ';';
    title.textContent = 'Tokenizer Comparison';
    root.appendChild(title);

    // Text selector
    var selectorRow = document.createElement('div');
    selectorRow.style.cssText = 'margin-bottom: 16px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center;';

    var label = document.createElement('span');
    label.style.cssText = 'font-size: 14px; color: ' + colors.textMuted + ';';
    label.textContent = 'Input text:';
    selectorRow.appendChild(label);

    var select = document.createElement('select');
    select.style.cssText = 'padding: 4px 8px; border-radius: 4px; border: 1px solid ' +
      colors.controlBorder + '; background: ' + colors.controlBg +
      '; color: ' + colors.text + '; font-size: 14px; flex: 1; min-width: 200px;';
    EXAMPLE_TEXTS.forEach(function (text, i) {
      var opt = document.createElement('option');
      opt.value = i;
      opt.textContent = text;
      select.appendChild(opt);
    });
    // Custom option
    var customOpt = document.createElement('option');
    customOpt.value = 'custom';
    customOpt.textContent = '(type your own...)';
    select.appendChild(customOpt);
    selectorRow.appendChild(select);

    root.appendChild(selectorRow);

    // Custom text input (hidden initially)
    var customInput = document.createElement('input');
    customInput.type = 'text';
    customInput.placeholder = 'Type your own text here...';
    customInput.style.cssText = 'display: none; width: 100%; padding: 8px; margin-bottom: 16px; ' +
      'border-radius: 4px; border: 1px solid ' + colors.controlBorder +
      '; background: ' + colors.controlBg + '; color: ' + colors.text +
      '; font-size: 14px; box-sizing: border-box;';
    root.appendChild(customInput);

    // Results container
    var resultsDiv = document.createElement('div');
    root.appendChild(resultsDiv);

    function update() {
      var text;
      if (select.value === 'custom') {
        text = customInput.value || 'Type something above...';
        customInput.style.display = 'block';
      } else {
        text = EXAMPLE_TEXTS[parseInt(select.value)];
        customInput.style.display = 'none';
      }

      resultsDiv.innerHTML = '';
      var results = [];

      Object.keys(TOKENIZERS).forEach(function (name) {
        var tok = TOKENIZERS[name];
        var tokens = tok.fn(text);
        results.push({ name: name, tokens: tokens });

        // Description
        var desc = document.createElement('div');
        desc.style.cssText = 'font-size: 12px; color: ' + colors.textMuted +
          '; margin-bottom: 4px; font-style: italic;';
        desc.textContent = tok.desc;
        resultsDiv.appendChild(desc);

        renderTokens(resultsDiv, tokens, colors, name);
      });

      renderStats(resultsDiv, results, colors);
    }

    select.addEventListener('change', update);
    customInput.addEventListener('input', update);
    update();
  }

  // --- Bootstrap ---

  function bootstrap() {
    var containers = document.querySelectorAll('.tokenizer-viz');
    containers.forEach(function (el) {
      initViz(el);
    });
  }

  // Theme change observer
  var observer = new MutationObserver(function () {
    var containers = document.querySelectorAll('.tokenizer-viz');
    containers.forEach(function (el) {
      initViz(el);
    });
  });

  function startObserver() {
    observer.observe(document.body, {
      attributes: true,
      attributeFilter: ['data-md-color-scheme'],
    });
  }

  // Wait for DOM and mkdocs-material navigation
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () {
      bootstrap();
      startObserver();
    });
  } else {
    bootstrap();
    startObserver();
  }

  // Re-init on mkdocs-material page navigation (instant loading)
  if (typeof document$ !== 'undefined') {
    document$.subscribe(function () {
      bootstrap();
    });
  }
})();
