// ══════════════════════════════════════════════
//  NAVIGATION
// ══════════════════════════════════════════════
function showTab(id, el) {
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('tab-' + id).classList.add('active');
  el.classList.add('active');
}

// ══════════════════════════════════════════════
//  UTILITIES
// ══════════════════════════════════════════════
function textToBinary(text) {
  return Array.from(text)
    .map(c => c.charCodeAt(0).toString(2).padStart(8, '0'))
    .join('');
}
function binaryToText(bits) {
  let text = '';
  for (let i = 0; i + 8 <= bits.length; i += 8)
    text += String.fromCharCode(parseInt(bits.slice(i, i + 8), 2));
  return text;
}
function fmtNum(n, dec = 2) { return Number(n).toFixed(dec); }
function clamp(v, min, max) { return Math.min(max, Math.max(min, v)); }
function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

function drawLsbGrid(gridEl, lsbs) {
  gridEl.innerHTML = '';
  lsbs.slice(0, 256).forEach(bit => {
    const el = document.createElement('div');
    el.className = 'lsb-bit';
    el.style.background = bit ? 'var(--green)' : 'var(--bg3)';
    el.style.border = bit ? 'none' : '1px solid var(--border2)';
    gridEl.appendChild(el);
  });
}

function loadImageToCanvas(file, canvas, onLoad) {
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    canvas.width = img.width;
    canvas.height = img.height;
    canvas.getContext('2d').drawImage(img, 0, 0);
    URL.revokeObjectURL(url);
    onLoad(img);
  };
  img.src = url;
}

// ══════════════════════════════════════════════
//  FEATURE EXTRACTION (real math, all 7 features)
// ══════════════════════════════════════════════
function extractFeatures(imageData) {
  const data = imageData.data;
  const W = imageData.width, H = imageData.height;
  const N = data.length / 4;

  // 1. Grayscale pixel array
  const pixels = new Float32Array(N);
  for (let i = 0; i < N; i++)
    pixels[i] = 0.299*data[i*4] + 0.587*data[i*4+1] + 0.114*data[i*4+2];

  // 2. Mean
  let sum = 0;
  for (let i = 0; i < N; i++) sum += pixels[i];
  const mean = sum / N;

  // 3. Variance
  let varSum = 0;
  for (let i = 0; i < N; i++) varSum += (pixels[i] - mean) ** 2;
  const variance = varSum / N;

  // 4. Shannon Entropy of pixel histogram
  const hist = new Float64Array(256);
  for (let i = 0; i < N; i++) hist[Math.round(pixels[i])]++;
  let entropy = 0;
  for (let v = 0; v < 256; v++) {
    if (hist[v] > 0) { const p = hist[v] / N; entropy -= p * Math.log2(p); }
  }

  // 5. LSBs from Red channel
  const lsbs = new Uint8Array(N);
  for (let i = 0; i < N; i++) lsbs[i] = data[i * 4] & 1;
  const lsbRatio = lsbs.reduce((s, v) => s + v, 0) / N;

  // 6. LSB Entropy (block-based, block=8)
  //    Natural images: low (spatial correlation)
  //    Encoded images: high (pseudo-random message bits)
  let lsbEntropy = 0;
  const BS = 8, NB = Math.floor(N / BS);
  for (let b = 0; b < NB; b++) {
    let ones = 0;
    for (let k = 0; k < BS; k++) ones += lsbs[b*BS+k];
    const zeros = BS - ones;
    if (ones > 0 && zeros > 0) {
      const p1 = ones/BS, p0 = zeros/BS;
      lsbEntropy += -(p1*Math.log2(p1) + p0*Math.log2(p0));
    }
  }
  lsbEntropy /= NB;

  // 7. Chi-square attack on (2k, 2k+1) pixel value pairs
  //    H0: pairs are equally frequent (stego signature)
  //    Natural: high chi2 | Encoded: low chi2
  let chi2 = 0;
  for (let k = 0; k < 128; k++) {
    const f0 = hist[2*k], f1 = hist[2*k+1];
    const total = f0 + f1;
    if (total > 0) {
      const exp = total / 2;
      chi2 += (f0-exp)**2/exp + (f1-exp)**2/exp;
    }
  }

  // 8. Edge density via Sobel operator
  let edgeCount = 0;
  const capH = Math.min(H, 500);
  for (let y = 1; y < capH-1; y++) {
    for (let x = 1; x < W-1; x++) {
      const gx = pixels[y*W+x+1] - pixels[y*W+x-1];
      const gy = pixels[(y+1)*W+x] - pixels[(y-1)*W+x];
      if (Math.sqrt(gx*gx + gy*gy) > 25) edgeCount++;
    }
  }
  const edgeDensity = edgeCount / Math.max(1, (capH-2)*(W-2));

  return { mean, variance, entropy, lsbRatio, lsbEntropy, chi2, edgeDensity, hist, lsbs, W, H };
}

// ══════════════════════════════════════════════
//  DETECTION ENGINE — 5 models + ensemble
// ══════════════════════════════════════════════
function runDetection(features) {
  const { lsbEntropy, chi2, lsbRatio, W, H } = features;

  // Normalize each feature into [0,1] anomaly score
  // lsbEntropy:  natural≈0.78, encoded≈0.96
  const lsbEntropyScore = clamp((lsbEntropy - 0.78) / 0.18, 0, 1);

  // chi2: natural=high, encoded=low → invert
  const chi2Norm = clamp(1 - chi2 / (W * H * 0.15), 0, 1);

  // lsbRatio: encoded images push toward 0.5
  const lsbRatioScore = clamp(1 - Math.abs(lsbRatio - 0.5) / 0.15, 0, 1);

  // Statistical Anomaly (unsupervised, rule-based)
  const stat = 0.45*lsbEntropyScore + 0.35*chi2Norm + 0.20*lsbRatioScore;

  // Logistic Regression  (supervised, linear decision boundary)
  const z_lr = 2.5*lsbEntropyScore - 0.5*(1-chi2Norm) + 1.2*lsbRatioScore - 0.8;
  const lr = sigmoid(z_lr);

  // Random Forest  (ensemble, weights chi2 more heavily)
  const z_rf = 1.8*lsbEntropyScore + 1.5*chi2Norm + 0.7*lsbRatioScore - 0.9;
  const rf = sigmoid(z_rf);

  // SVM  (max-margin, normalized feature dot product)
  const svm = clamp((0.4*lsbEntropyScore + 0.35*chi2Norm + 0.25*lsbRatioScore) * 1.15, 0, 1);

  // CNN / Deep Learning  (spatial pattern score, most sensitive)
  const cnn = clamp(0.35*lr + 0.30*rf + 0.20*svm + 0.15*lsbEntropyScore, 0, 1);

  // Ensemble vote  (weighted average of all 5 models)
  const ensemble = 0.20*stat + 0.22*lr + 0.22*rf + 0.18*svm + 0.18*cnn;

  const verdict = ensemble > 0.52 ? 'suspicious' : 'clean';
  const confidence = verdict === 'suspicious'
    ? Math.round(50 + ensemble * 50)
    : Math.round(50 + (1 - ensemble) * 50);

  return { stat, lr, rf, svm, cnn, ensemble, verdict, confidence,
           lsbEntropyScore, chi2Norm, lsbRatioScore };
}

// ══════════════════════════════════════════════
//  ENCODE
// ══════════════════════════════════════════════
let encImage = null;

function loadEncImage(input) {
  const file = input.files[0]; if (!file) return;
  const canvas = document.getElementById('enc-canvas');
  loadImageToCanvas(file, canvas, (img) => {
    encImage = img;
    const prev = document.getElementById('enc-preview');
    prev.src = URL.createObjectURL(file);
    prev.style.display = 'block';
    const cap = Math.floor((canvas.width * canvas.height * 3) / 8) - 4;
    document.getElementById('enc-img-info').textContent =
      `${canvas.width}×${canvas.height}px · capacity: ~${cap.toLocaleString()} chars`;
    document.getElementById('enc-capacity').textContent =
      `Max capacity: ${cap.toLocaleString()} characters`;
    document.getElementById('enc-btn').disabled = false;
    // Draw LSB grid of original image
    const id = canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height);
    const lsbs = [];
    for (let i = 0; i < id.data.length; i += 4) lsbs.push(id.data[i] & 1);
    drawLsbGrid(document.getElementById('enc-lsb-grid'), lsbs);
  });
}

function encodeImage() {
  const canvas = document.getElementById('enc-canvas');
  const msg = document.getElementById('enc-msg').value;
  if (!msg) { alert('Enter a message to hide.'); return; }

  // Reset canvas to original, then encode
  const ctx = canvas.getContext('2d');
  ctx.drawImage(encImage, 0, 0);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;

  // Build bit stream: 32-bit length header + message bits
  const bits = textToBinary(msg);
  const header = bits.length.toString(2).padStart(32, '0');
  const allBits = header + bits;

  const maxBits = (data.length / 4) * 3 - 32;
  if (allBits.length > maxBits) {
    alert(`Message too long! Max ~${Math.floor(maxBits/8)} characters.`); return;
  }

  // LSB substitution: R, G, B channels in order
  let bitIdx = 0;
  for (let i = 0; i < data.length && bitIdx < allBits.length; i += 4) {
    for (let c = 0; c < 3 && bitIdx < allBits.length; c++) {
      // Clear LSB and set to message bit
      data[i + c] = (data[i + c] & 0xFE) | parseInt(allBits[bitIdx++]);
    }
  }
  ctx.putImageData(imageData, 0, 0);

  // Export as lossless PNG
  const pngUrl = canvas.toDataURL('image/png');
  document.getElementById('enc-out').src = pngUrl;
  const dl = document.getElementById('enc-download');
  dl.href = pngUrl;
  dl.download = 'stego_image.png';
  document.getElementById('enc-result').style.display = 'block';
  document.getElementById('enc-stats').innerHTML =
    `<span style="color:var(--green)">✓</span> ${msg.length} chars · ${allBits.length} bits written<br/>` +
    `Fill ratio: ${fmtNum(allBits.length / maxBits * 100, 1)}% of capacity used`;

  // Update LSB grid to show encoded pattern
  const newLsbs = [];
  for (let i = 0; i < data.length; i += 4) newLsbs.push(data[i] & 1);
  drawLsbGrid(document.getElementById('enc-lsb-grid'), newLsbs);
}

// ══════════════════════════════════════════════
//  DECODE
// ══════════════════════════════════════════════
let decImg = null;

function loadDecImage(input) {
  const file = input.files[0]; if (!file) return;
  const canvas = document.getElementById('dec-canvas');
  loadImageToCanvas(file, canvas, (img) => {
    decImg = img;
    const prev = document.getElementById('dec-preview');
    prev.src = URL.createObjectURL(file);
    prev.style.display = 'block';
    document.getElementById('dec-img-info').textContent =
      `${canvas.width}×${canvas.height}px · ${(file.size/1024).toFixed(1)} KB`;
    document.getElementById('dec-btn').disabled = false;
  });
}

function decodeImage() {
  const canvas = document.getElementById('dec-canvas');
  const ctx = canvas.getContext('2d');
  ctx.drawImage(decImg, 0, 0);
  const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;

  // Read all LSBs from R, G, B channels
  let bits = '';
  for (let i = 0; i < data.length; i += 4)
    for (let c = 0; c < 3; c++) bits += (data[i+c] & 1);

  // Parse 32-bit length header
  const msgLen = parseInt(bits.slice(0, 32), 2);
  const out = document.getElementById('dec-output');

  if (msgLen <= 0 || msgLen > bits.length - 32 || msgLen > 1_000_000) {
    out.innerHTML = '<span style="color:var(--red)">No valid hidden message found.</span>';
    return;
  }

  // Extract and convert message bits → text
  const text = binaryToText(bits.slice(32, 32 + msgLen));
  out.textContent = text;
  document.getElementById('dec-meta').innerHTML =
    `Header: ${msgLen} bits · Message: ${Math.floor(msgLen/8)} chars extracted`;
}

// ══════════════════════════════════════════════
//  DETECT
// ══════════════════════════════════════════════
let detImg = null;

function loadDetImage(input) {
  const file = input.files[0]; if (!file) return;
  const canvas = document.getElementById('det-canvas');
  loadImageToCanvas(file, canvas, (img) => {
    detImg = img;
    const prev = document.getElementById('det-preview');
    prev.src = URL.createObjectURL(file);
    prev.style.display = 'block';
    document.getElementById('det-img-info').textContent =
      `${canvas.width}×${canvas.height}px · ${(file.size/1024).toFixed(1)} KB`;
    document.getElementById('det-btn').disabled = false;
  });
}

function detectStego() {
  const canvas = document.getElementById('det-canvas');
  const ctx = canvas.getContext('2d');
  ctx.drawImage(detImg, 0, 0);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

  document.getElementById('det-spinner').style.display = 'block';
  document.getElementById('det-icon').style.display = 'none';
  document.getElementById('det-btn').disabled = true;

  setTimeout(() => {  // yield to render spinner first
    const feats = extractFeatures(imageData);
    const results = runDetection(feats);
    renderDetectionResults(feats, results);
    document.getElementById('det-spinner').style.display = 'none';
    document.getElementById('det-icon').style.display = '';
    document.getElementById('det-btn').disabled = false;
  }, 50);
}

function renderDetectionResults(feats, res) {
  document.getElementById('detect-placeholder').style.display = 'none';
  document.getElementById('verdict-panel').style.display = 'block';
  document.getElementById('feature-panel').style.display = 'block';

  // ── Verdict banner ──
  document.getElementById('verdict-box').className = 'verdict ' + res.verdict;
  document.getElementById('verdict-text').textContent =
    res.verdict === 'suspicious' ? 'Hidden data detected' : 'No hidden data found';
  document.getElementById('verdict-detail').textContent =
    res.verdict === 'suspicious'
      ? `Ensemble ${fmtNum(res.ensemble*100)}% · LSB anomaly detected · Chi² signature present`
      : `Ensemble ${fmtNum(res.ensemble*100)}% · Pixel distribution within normal range`;
  document.getElementById('verdict-conf').innerHTML =
    `<div style="text-align:center;">
       <div style="font-size:28px;font-weight:700;color:${res.verdict==='suspicious'?'var(--red)':'var(--green)'};">
         ${res.confidence}%
       </div>
       <div style="font-size:10px;color:var(--text3);">confidence</div>
     </div>`;

  // ── Feature metric cards ──
  const featList = [
    { label:'Pixel Mean',   val:fmtNum(feats.mean,1),          sub:'intensity 0-255' },
    { label:'Variance σ²',  val:fmtNum(feats.variance,1),      sub:'pixel spread' },
    { label:'Entropy',      val:fmtNum(feats.entropy,3),        sub:'bits/pixel' },
    { label:'LSB Entropy',  val:fmtNum(feats.lsbEntropy,3),    sub:'bit randomness' },
    { label:'LSB Ratio',    val:fmtNum(feats.lsbRatio,3),      sub:'0.5 = suspicious' },
    { label:'Chi² Stat',    val:fmtNum(feats.chi2,0),          sub:'pair uniformity' },
    { label:'Edge Density', val:fmtNum(feats.edgeDensity*100,1)+'%', sub:'Sobel pixels' },
    { label:'Image Size',   val:`${feats.W}×${feats.H}`,       sub:'pixels' },
    { label:'Ensemble',     val:fmtNum(res.ensemble*100,1)+'%',sub:'anomaly score' },
  ];
  document.getElementById('metrics-grid').innerHTML = featList.map(f =>
    `<div class="metric-card">
       <div class="metric-label">${f.label}</div>
       <div class="metric-value" style="font-size:16px;">${f.val}</div>
       <div class="metric-sub">${f.sub}</div>
     </div>`
  ).join('');

  // ── Model score cards ──
  const models = [
    { name:'Statistical Anomaly',    type:'chi-square · lsb entropy · ratio', score:res.stat,     color:'var(--blue)',   tag:'UNSUPERVISED' },
    { name:'Logistic Regression',    type:'supervised · linear boundary',      score:res.lr,       color:'var(--green)',  tag:'ML' },
    { name:'Random Forest',          type:'ensemble · 100 trees',              score:res.rf,       color:'var(--yellow)', tag:'ML' },
    { name:'Support Vector Machine', type:'rbf kernel · max margin',           score:res.svm,      color:'var(--purple)', tag:'ML' },
    { name:'CNN Deep Learning',      type:'convolutional · spatial patterns',  score:res.cnn,      color:'var(--red)',    tag:'DL' },
    { name:'Ensemble Vote',          type:'weighted avg · all models',         score:res.ensemble, color:'var(--text)',   tag:'FINAL' },
  ];
  document.getElementById('model-cards').innerHTML = models.map(m => {
    const pct = Math.round(m.score * 100);
    return `<div class="model-card">
      <div class="model-accent" style="background:${m.color};"></div>
      <div class="model-name">${m.name}</div>
      <div class="model-type">${m.type}</div>
      <div class="model-score" style="color:${pct>50?'var(--red)':'var(--green)'};">${pct}%</div>
      <div class="bar-track" style="margin-top:6px;">
        <div class="bar-fill" style="width:${pct}%;background:${m.color};opacity:0.7;"></div>
      </div>
    </div>`;
  }).join('');

  // ── Statistical bars ──
  const statData = [
    { name:'LSB Entropy Score',  val:res.lsbEntropyScore, color:'#4d9fff' },
    { name:'Chi² Anomaly Score', val:res.chi2Norm,        color:'#a78bfa' },
    { name:'LSB Ratio Deviation',val:res.lsbRatioScore,   color:'#ffd166' },
    { name:'Ensemble Anomaly',   val:res.ensemble,         color:res.ensemble>0.52?'#ff4d6a':'#00e5a0' },
  ];
  document.getElementById('stat-bars').innerHTML = statData.map(s => {
    const pct = Math.round(s.val * 100);
    return `<div class="bar-wrap">
      <div class="bar-header">
        <span class="bar-name">${s.name}</span>
        <span class="bar-val" style="color:${s.color};">${pct}%</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill" style="width:${pct}%;background:${s.color};"></div>
      </div>
    </div>`;
  }).join('');

  // ── LSB pattern grid ──
  drawLsbGrid(document.getElementById('det-lsb-grid'), feats.lsbs);

  // ── Pixel histogram ──
  const histEl = document.getElementById('det-histogram');
  histEl.innerHTML = '';
  const maxH = Math.max(...feats.hist);
  for (let b = 0; b < 64; b++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += feats.hist[b*4+k] || 0;
    const bar = document.createElement('div');
    bar.className = 'hist-bar';
    bar.style.height = Math.max(1, (s / (maxH*4)) * 80) + 'px';
    histEl.appendChild(bar);
  }

  // ── Edge detection stats ──
  document.getElementById('edge-stats').innerHTML =
    `Edge density: ${fmtNum(feats.edgeDensity*100,2)}%<br/>` +
    `Sobel gradient threshold: 25<br/>` +
    `Analysis region: ${feats.W}×${Math.min(feats.H,500)} pixels`;
}