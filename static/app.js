document.addEventListener('DOMContentLoaded', () => {
  const form = document.querySelector('form[action="/"]');
  const analyzeBtn = form?.querySelector('button[type="submit"]');
  const clearBtn = document.getElementById('clear-btn');
  const copyBtn = document.getElementById('copy-btn');
  const emailTA = document.getElementById('email');

  if (form && analyzeBtn) {
    form.addEventListener('submit', () => {
      analyzeBtn.dataset.prev = analyzeBtn.textContent || '';
      analyzeBtn.textContent = 'Analyzing…';
      analyzeBtn.classList.add('is-loading');
      analyzeBtn.disabled = true;
    });
  }

  if (clearBtn && emailTA) {
    clearBtn.addEventListener('click', () => {
      emailTA.value = '';
      emailTA.focus();
    });
  }

  if (copyBtn) {
    copyBtn.addEventListener('click', async () => {
      const result = document.getElementById('result-block');
      if (!result) return;
      const text = result.innerText;
      try {
        await navigator.clipboard.writeText(text);
        copyBtn.textContent = 'Copied!';
        setTimeout(() => (copyBtn.textContent = 'Copy result'), 1200);
      } catch (e) {
        console.warn('Copy failed', e);
      }
    });
  }
});


