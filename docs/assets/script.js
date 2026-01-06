(function () {
  function getCopyText(root) {
    var code = root.querySelector('pre code');
    return code ? code.textContent : '';
  }

  function attachCopyHandlers() {
    var blocks = document.querySelectorAll('[data-copy]');
    for (var i = 0; i < blocks.length; i++) {
      (function (block) {
        var btn = block.querySelector('.codeblock__copy');
        if (!btn) return;

        btn.addEventListener('click', async function () {
          var text = getCopyText(block);
          if (!text) return;

          try {
            await navigator.clipboard.writeText(text);
            var old = btn.textContent;
            btn.textContent = 'Copied';
            btn.disabled = true;
            setTimeout(function () {
              btn.textContent = old;
              btn.disabled = false;
            }, 900);
          } catch (e) {
            // Fallback: selecciona el texto
            var code = block.querySelector('pre');
            if (!code) return;
            var range = document.createRange();
            range.selectNodeContents(code);
            var sel = window.getSelection();
            sel.removeAllRanges();
            sel.addRange(range);
          }
        });
      })(blocks[i]);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', attachCopyHandlers);
  } else {
    attachCopyHandlers();
  }
})();
