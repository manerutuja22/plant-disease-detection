// Frontend logic for image preview, drag-drop, prediction, and displaying results
(function(){
  const $ = (id)=>document.getElementById(id);
  const imageInput = $('imageInput');
  const dropArea = $('dropArea');
  const previewContainer = $('previewContainer');
  const previewImage = $('previewImage');
  const predictBtn = $('predictBtn');
  const clearBtn = $('clearBtn');
  const resultCard = $('resultCard');
  const resultLabel = $('resultLabel');
  const resultConfidence = $('resultConfidence');
  const resultInfo = $('resultInfo');
  const statusEl = $('status');
  const spinner = $('spinner');
  const top3 = $('top3');
  const top3List = $('top3List');
  const quality = $('quality');
  const qualityList = $('qualityList');
  const themeToggle = $('themeToggle');

  let previewUrl = null;

  function resetUI(){
    if(previewUrl){ URL.revokeObjectURL(previewUrl); previewUrl = null; }
    imageInput.value = '';
    previewImage.removeAttribute('src');
    previewContainer.classList.add('hidden');
    predictBtn.disabled = true;
    resultCard.classList.add('hidden');
    resultLabel.textContent = '—';
    resultConfidence.textContent = '—';
    resultInfo.textContent = '—';
    statusEl.textContent = '';
    top3.classList.add('hidden');
    quality.classList.add('hidden');
    top3List.innerHTML = '';
    qualityList.innerHTML = '';
  }

  function showSpinner(show){
    spinner.classList.toggle('hidden', !show);
  }

  function setPreview(file){
    if(!file) return;
    if(previewUrl){ URL.revokeObjectURL(previewUrl); }
    previewUrl = URL.createObjectURL(file);
    previewImage.src = previewUrl;
    previewContainer.classList.remove('hidden');
    predictBtn.disabled = false;
    resultCard.classList.add('hidden');
    statusEl.textContent = '';
  }

  imageInput.addEventListener('change', () => setPreview(imageInput.files && imageInput.files[0]));

  // Drag and drop handlers
  if(dropArea){
    ['dragenter','dragover'].forEach(evt => dropArea.addEventListener(evt,(e)=>{e.preventDefault();dropArea.classList.add('hover');}));
    ['dragleave','drop'].forEach(evt => dropArea.addEventListener(evt,(e)=>{e.preventDefault();dropArea.classList.remove('hover');}));
    dropArea.addEventListener('drop', (e)=>{
      const file = e.dataTransfer.files && e.dataTransfer.files[0];
      if(file) setPreview(file);
    });
  }

  clearBtn.addEventListener('click', resetUI);

  predictBtn.addEventListener('click', async () => {
    const file = (imageInput.files && imageInput.files[0]) || (previewUrl && await fetch(previewUrl).then(r=>r.blob()).then(b=>new File([b],'upload.jpg')));
    if(!file){ statusEl.textContent = 'Please select an image first.'; return; }

    predictBtn.disabled = true;
    statusEl.textContent = 'Predicting…';
    showSpinner(true);

    try{
      const form = new FormData();
      form.append('image', file, file.name || 'upload.jpg');

      const res = await fetch('/predict', { method: 'POST', body: form });
      if(!res.ok){
        const txt = await res.text();
        throw new Error(`Server error (${res.status}): ${txt}`);
      }
      const data = await res.json();
      // Expected: { label, confidence, info, top3?, quality? }
      resultLabel.textContent = data.label || 'Unknown';
      const pct = (typeof data.confidence === 'number' ? data.confidence : 0) * 100;
      resultConfidence.textContent = `${pct.toFixed(1)}% confidence`;
      resultInfo.textContent = data.info || '—';
      resultCard.classList.remove('hidden');

      // Render top3
      if(Array.isArray(data.top3) && data.top3.length){
        top3List.innerHTML = '';
        data.top3.forEach(item => {
          const li = document.createElement('div');
          li.className = 'top3-item';
          const name = document.createElement('div');
          name.className = 'name';
          name.textContent = item.label;
          const pct = document.createElement('div');
          pct.className = 'pct';
          pct.textContent = `${(item.confidence*100).toFixed(1)}%`;
          const bar = document.createElement('div');
          bar.className = 'bar';
          const fill = document.createElement('span');
          fill.style.width = `${Math.max(3, item.confidence*100)}%`;
          bar.appendChild(fill);
          const row = document.createElement('div');
          row.style.gridColumn = '1 / -1';
          row.appendChild(bar);
          li.appendChild(name);
          li.appendChild(pct);
          li.appendChild(row);
          top3List.appendChild(li);
        });
        top3.classList.remove('hidden');
      }else{
        top3.classList.add('hidden');
      }

      // Render quality
      if(data.quality){
        qualityList.innerHTML = '';
        const items = [
          `Focus: ${data.quality.blur_verdict} (score ${data.quality.blur_score.toFixed(1)})`,
          `Exposure: ${data.quality.exposure_verdict} (avg ${data.quality.brightness.toFixed(0)})`,
          data.quality.tips || ''
        ].filter(Boolean);
        items.forEach(t=>{ const li=document.createElement('li'); li.textContent=t; qualityList.appendChild(li);});
        quality.classList.remove('hidden');
      }else{
        quality.classList.add('hidden');
      }

      statusEl.textContent = '';
    }catch(err){
      console.error(err);
      statusEl.textContent = err.message || 'Prediction failed.';
      resultCard.classList.add('hidden');
    }finally{
      predictBtn.disabled = false;
      showSpinner(false);
    }
  });

  // Theme toggle
  try{
    const stored = localStorage.getItem('theme');
    if(stored === 'dark'){ document.body.classList.add('theme-dark'); themeToggle.checked = true; }
    themeToggle.addEventListener('change', () => {
      document.body.classList.toggle('theme-dark', themeToggle.checked);
      localStorage.setItem('theme', themeToggle.checked ? 'dark' : 'light');
    });
  }catch{}

  // Initialize
  resetUI();
})();
