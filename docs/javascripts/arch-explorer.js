/**
 * LMT Architecture Explorer
 *
 * Interactive visualization showing the building blocks of each model
 * architecture. Click on a model name to see its transformer block
 * composition.
 *
 * Usage: Add <div class="arch-explorer"></div> to any markdown page.
 */
(function () {
  'use strict';

  function isDarkTheme() {
    return document.body.getAttribute('data-md-color-scheme') === 'slate';
  }

  function getColors() {
    var dark = isDarkTheme();
    return {
      bg: dark ? '#1e1e2e' : '#ffffff',
      text: dark ? '#cdd6f4' : '#1a1a2e',
      textMuted: dark ? '#6c7086' : '#888888',
      border: dark ? '#45475a' : '#e0e0e0',
      btnBg: dark ? '#313244' : '#f5f5f5',
      btnActive: dark ? '#cba6f7' : '#6a1b9a',
      btnActiveText: dark ? '#1e1e2e' : '#ffffff',
      // Component colors
      attn: dark ? '#89b4fa' : '#1565c0',
      ffn: dark ? '#a6e3a1' : '#2e7d32',
      norm: dark ? '#f9e2af' : '#f57f17',
      pos: dark ? '#f38ba8' : '#c62828',
      moe: dark ? '#cba6f7' : '#6a1b9a',
      embed: dark ? '#94e2d5' : '#00695c',
    };
  }

  var MODELS = {
    GPT: {
      desc: 'The original decoder-only transformer',
      params: '~40K',
      blocks: [
        { type: 'embed', label: 'Token Embedding + Learned Pos' },
        { type: 'norm', label: 'LayerNorm (pre-norm)' },
        { type: 'attn', label: 'Multi-Head Attention (MHA)' },
        { type: 'norm', label: 'LayerNorm (pre-norm)' },
        { type: 'ffn', label: 'FFN (GELU activation)' },
        { type: 'embed', label: 'LM Head (tied weights)' },
      ],
    },
    LLaMA: {
      desc: 'Modern open-source standard',
      params: '~38K',
      blocks: [
        { type: 'embed', label: 'Token Embedding' },
        { type: 'norm', label: 'RMSNorm (pre-norm)' },
        { type: 'attn', label: 'Grouped Query Attention (GQA)' },
        { type: 'pos', label: 'RoPE (rotary position)' },
        { type: 'norm', label: 'RMSNorm (pre-norm)' },
        { type: 'ffn', label: 'SwiGLU FFN' },
        { type: 'embed', label: 'LM Head (tied weights)' },
      ],
    },
    Qwen3: {
      desc: 'LLaMA-style with QK-norm',
      params: '~38K',
      blocks: [
        { type: 'embed', label: 'Token Embedding' },
        { type: 'norm', label: 'RMSNorm (pre-norm)' },
        { type: 'attn', label: 'GQA + QK-Norm' },
        { type: 'pos', label: 'RoPE (rotary position)' },
        { type: 'norm', label: 'RMSNorm (pre-norm)' },
        { type: 'ffn', label: 'SwiGLU FFN' },
        { type: 'embed', label: 'LM Head (tied weights)' },
      ],
    },
    Gemma: {
      desc: 'Google\'s efficient design',
      params: '~42K',
      blocks: [
        { type: 'embed', label: 'Token Embedding (scaled)' },
        { type: 'norm', label: 'RMSNorm (pre-norm)' },
        { type: 'attn', label: 'Multi-Head Attention' },
        { type: 'pos', label: 'RoPE (rotary position)' },
        { type: 'norm', label: 'RMSNorm (post-attn)' },
        { type: 'norm', label: 'RMSNorm (pre-FFN)' },
        { type: 'ffn', label: 'GeGLU FFN' },
        { type: 'norm', label: 'RMSNorm (post-FFN)' },
        { type: 'embed', label: 'LM Head (tied weights)' },
      ],
    },
    Mixtral: {
      desc: 'Sparse Mixture-of-Experts',
      params: '~85K',
      blocks: [
        { type: 'embed', label: 'Token Embedding' },
        { type: 'norm', label: 'RMSNorm (pre-norm)' },
        { type: 'attn', label: 'GQA Attention' },
        { type: 'pos', label: 'RoPE (rotary position)' },
        { type: 'norm', label: 'RMSNorm (pre-norm)' },
        { type: 'moe', label: 'MoE: Top-2 of 4 Experts (SwiGLU)' },
        { type: 'embed', label: 'LM Head (tied weights)' },
      ],
    },
    Mamba: {
      desc: 'State Space Model (no attention)',
      params: '~35K',
      blocks: [
        { type: 'embed', label: 'Token Embedding' },
        { type: 'norm', label: 'RMSNorm (pre-norm)' },
        { type: 'attn', label: 'Mamba SSM Block' },
        { type: 'embed', label: 'LM Head (tied weights)' },
      ],
    },
  };

  function render(container) {
    var colors = getColors();
    container.innerHTML = '';
    container.style.fontFamily =
      '-apple-system, BlinkMacSystemFont, sans-serif';

    // Model selector buttons
    var btnRow = document.createElement('div');
    btnRow.style.cssText =
      'display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px;';
    container.appendChild(btnRow);

    // Detail panel
    var panel = document.createElement('div');
    panel.style.cssText =
      'border:1px solid ' + colors.border +
      ';border-radius:8px;padding:16px;min-height:200px;' +
      'background:' + colors.bg + ';';
    container.appendChild(panel);

    var activeModel = 'GPT';

    function showModel(name) {
      activeModel = name;
      var model = MODELS[name];
      var c = getColors(); // Re-fetch for theme changes

      // Update buttons
      var btns = btnRow.querySelectorAll('button');
      for (var i = 0; i < btns.length; i++) {
        var isActive = btns[i].textContent === name;
        btns[i].style.background = isActive ? c.btnActive : c.btnBg;
        btns[i].style.color = isActive ? c.btnActiveText : c.text;
        btns[i].style.fontWeight = isActive ? 'bold' : 'normal';
      }

      // Build block diagram
      var html = '<div style="margin-bottom:12px;">';
      html += '<strong style="font-size:16px;color:' + c.text + ';">' +
        name + '</strong>';
      html += '<span style="color:' + c.textMuted +
        ';margin-left:12px;font-size:13px;">' +
        model.desc + '</span>';
      html += '</div>';

      html += '<div style="display:flex;flex-direction:column;gap:4px;">';

      for (var j = 0; j < model.blocks.length; j++) {
        var b = model.blocks[j];
        var blockColor = c[b.type] || c.text;
        var arrow = j < model.blocks.length - 1
          ? '<div style="text-align:center;color:' + c.textMuted +
            ';font-size:14px;line-height:16px;">↓</div>'
          : '';

        html +=
          '<div style="background:' + blockColor + '18;' +
          'border-left:3px solid ' + blockColor + ';' +
          'padding:8px 12px;border-radius:4px;' +
          'color:' + c.text + ';font-size:13px;">' +
          '<span style="color:' + blockColor +
          ';font-weight:bold;margin-right:8px;">' +
          b.type.toUpperCase() + '</span>' +
          b.label + '</div>' + arrow;
      }

      html += '</div>';

      // Legend
      html += '<div style="margin-top:16px;display:flex;gap:16px;' +
        'flex-wrap:wrap;font-size:11px;color:' + c.textMuted + ';">';
      var types = [
        ['embed', 'Embedding'], ['norm', 'Normalization'],
        ['attn', 'Sequence Mixer'], ['ffn', 'Feed-Forward'],
        ['pos', 'Position'], ['moe', 'Mixture of Experts'],
      ];
      for (var k = 0; k < types.length; k++) {
        html += '<span>■ <span style="color:' + c[types[k][0]] +
          ';">' + types[k][1] + '</span></span>';
      }
      html += '</div>';

      panel.innerHTML = html;
      panel.style.background = c.bg;
      panel.style.borderColor = c.border;
    }

    // Create buttons
    var modelNames = Object.keys(MODELS);
    for (var m = 0; m < modelNames.length; m++) {
      var btn = document.createElement('button');
      btn.textContent = modelNames[m];
      btn.style.cssText =
        'padding:6px 14px;border:1px solid ' + colors.border +
        ';border-radius:16px;cursor:pointer;font-size:13px;' +
        'transition:all 0.15s;background:' + colors.btnBg +
        ';color:' + colors.text + ';';
      btn.addEventListener('click', (function (n) {
        return function () { showModel(n); };
      })(modelNames[m]));
      btnRow.appendChild(btn);
    }

    showModel(activeModel);
  }

  function initExplorers() {
    var containers = document.querySelectorAll('.arch-explorer');
    for (var i = 0; i < containers.length; i++) {
      render(containers[i]);
    }
  }

  // Theme change observer
  var observer = new MutationObserver(function (mutations) {
    for (var i = 0; i < mutations.length; i++) {
      if (mutations[i].attributeName === 'data-md-color-scheme') {
        initExplorers();
        break;
      }
    }
  });

  if (document.body) {
    observer.observe(document.body, { attributes: true });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initExplorers);
  } else {
    initExplorers();
  }

  if (typeof document$ !== 'undefined') {
    document$.subscribe(initExplorers);
  }
})();
