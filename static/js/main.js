/* ============================================================
   AdaIN Neural Style Transfer — Main JavaScript
   ============================================================ */

(function () {
  'use strict';

  // ── DOM References ──
  const contentZone     = document.getElementById('content-zone');
  const styleZone       = document.getElementById('style-zone');
  const contentInput    = document.getElementById('content-input');
  const styleInput      = document.getElementById('style-input');
  const alphaSlider     = document.getElementById('alpha-slider');
  const alphaValue      = document.getElementById('alpha-value');
  const generateBtn     = document.getElementById('generate-btn');
  const outputWrapper   = document.getElementById('output-wrapper');
  const outputImage     = document.getElementById('output-image');
  const outputEmpty     = document.getElementById('output-empty');
  const downloadBtn     = document.getElementById('download-btn');
  const loadingOverlay  = document.getElementById('loading-overlay');
  const toastContainer  = document.getElementById('toast-container');

  // ── State ──
  let contentFile = null;
  let styleFile   = null;

  // ── Toast System ──
  function showToast(message, type = 'error') {
    const toast = document.createElement('div');
    toast.className = `toast toast--${type}`;

    const icon = type === 'error'
      ? `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#f87171" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>`
      : `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2dd4bf" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>`;

    toast.innerHTML = icon + `<span>${message}</span>`;
    toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.classList.add('leaving');
      toast.addEventListener('animationend', () => toast.remove());
    }, 4000);
  }

  // ── Upload Zone Logic ──
  function setupUploadZone(zone, fileInput, type) {
    const placeholder = zone.querySelector('.upload-zone__placeholder');
    const previewContainer = zone.querySelector('.upload-zone__preview-container');
    const previewImg = zone.querySelector('.upload-zone__preview');
    const filenameEl = zone.querySelector('.upload-zone__filename');
    const removeBtn = zone.querySelector('.upload-zone__remove');

    // Click to browse
    zone.addEventListener('click', (e) => {
      if (e.target === removeBtn || removeBtn.contains(e.target)) return;
      fileInput.click();
    });

    // Drag & drop events
    ['dragenter', 'dragover'].forEach(event => {
      zone.addEventListener(event, (e) => {
        e.preventDefault();
        e.stopPropagation();
        zone.classList.add('drag-over');
      });
    });

    ['dragleave', 'drop'].forEach(event => {
      zone.addEventListener(event, (e) => {
        e.preventDefault();
        e.stopPropagation();
        zone.classList.remove('drag-over');
      });
    });

    zone.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        handleFile(files[0], type, previewImg, filenameEl, zone);
      }
    });

    // File input change
    fileInput.addEventListener('change', () => {
      if (fileInput.files.length > 0) {
        handleFile(fileInput.files[0], type, previewImg, filenameEl, zone);
      }
    });

    // Remove button
    removeBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      if (type === 'content') contentFile = null;
      else styleFile = null;
      zone.classList.remove('has-image');
      previewImg.src = '';
      filenameEl.textContent = '';
      fileInput.value = '';
    });
  }

  function handleFile(file, type, previewImg, filenameEl, zone) {
    // Validate
    const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
    if (!validTypes.includes(file.type)) {
      showToast('Please upload a valid image (JPG, PNG, or WebP).');
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      showToast('Image is too large. Max size is 20 MB.');
      return;
    }

    if (type === 'content') contentFile = file;
    else styleFile = file;

    // Preview
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      filenameEl.textContent = file.name;
      zone.classList.add('has-image');
    };
    reader.readAsDataURL(file);
  }

  // ── Alpha Slider ──
  alphaSlider.addEventListener('input', () => {
    alphaValue.textContent = parseFloat(alphaSlider.value).toFixed(1);
  });

  // ── Generate ──
  generateBtn.addEventListener('click', async () => {
    if (!contentFile) {
      showToast('Please upload a content image first.');
      return;
    }
    if (!styleFile) {
      showToast('Please upload a style image first.');
      return;
    }

    const formData = new FormData();
    formData.append('content', contentFile);
    formData.append('style', styleFile);
    formData.append('alpha', alphaSlider.value);

    // Show loading
    loadingOverlay.classList.add('active');
    generateBtn.disabled = true;

    try {
      const response = await fetch('/transfer', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.error || `Server error (${response.status})`);
      }

      const data = await response.blob();

      // Show output
      const imageUrl = URL.createObjectURL(data);
      outputImage.src = imageUrl;
      outputWrapper.classList.add('visible');
      outputEmpty.style.display = 'none';

      // Download button
      downloadBtn.onclick = () => {
        const a = document.createElement('a');
        a.href = imageUrl;
        a.download = 'stylized_output.jpg';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
      };

      showToast('Style transfer complete!', 'success');

      // Scroll to output
      document.getElementById('output-section').scrollIntoView({ behavior: 'smooth', block: 'center' });

    } catch (error) {
      showToast(error.message || 'Something went wrong. Please try again.');
    } finally {
      loadingOverlay.classList.remove('active');
      generateBtn.disabled = false;
    }
  });

  // ── Init ──
  setupUploadZone(contentZone, contentInput, 'content');
  setupUploadZone(styleZone, styleInput, 'style');

})();
