import './style.css';
import * as ort from 'onnxruntime-web';

const INPUT_SIZE = 640;

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <section id="center">
    <h1>YOLO26m Object Detection</h1>

    <p id="about">
      Object detection running entirely in your browser via
      <a href="https://onnxruntime.ai/docs/get-started/with-javascript/web.html" target="_blank" rel="noopener">ONNX Web Runtime</a>.
      When using a local file, your image is <strong>never sent to any server</strong> — all inference happens on your device.
      Model: <a href="https://docs.ultralytics.com/models/yolo26/" target="_blank" rel="noopener">Ultralytics YOLO26m</a>.
    </p>

    <div id="input-area">
      <div id="file-section">
        <h2>Local File</h2>
        <input type="file" accept="image/*" id="fileInput" />
      </div>
      <div id="iiif-section">
        <h2>IIIF Manifest (v2)</h2>
        <div id="iiif-url-row">
          <input type="text" id="manifestUrl" placeholder="Input Manifest URL..."
            value="https://dl.ndl.go.jp/api/iiif/3459985/manifest.json" />
          <button id="loadManifestBtn">Load</button>
        </div>
      </div>
    </div>

    <div id="main-area">
      <canvas id="canvas"></canvas>
      <div id="controls" style="display:none;">
        <div id="ep-selector">
          <span>Backend</span>
          <label><input type="radio" name="ep" value="cpu" checked /> CPU</label>
          <label id="webgpu-label"><input type="radio" name="ep" value="webgpu" /> WebGPU</label>
        </div>
        <label id="threshold-label-wrap">
          Threshold: <span id="thresholdLabel">0.20</span>
          <input type="range" id="thresholdSlider" min="0.05" max="0.95" step="0.05" value="0.20" />
        </label>
        <button id="detectBtn">Start Detection</button>
        <span id="inferenceTime"></span>
      </div>
    </div>

    <div id="thumbnail-grid"></div>

    <canvas id="for_prediction" style="display:none;"></canvas>
  </section>
`;

// --- Elements ---
const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const ctx = canvas.getContext('2d')!;
const predict_canvas = document.getElementById('for_prediction') as HTMLCanvasElement;
const predict_ctx = predict_canvas.getContext('2d')!;
const controls = document.getElementById('controls') as HTMLDivElement;
const thresholdSlider = document.getElementById('thresholdSlider') as HTMLInputElement;
const thresholdLabel = document.getElementById('thresholdLabel') as HTMLSpanElement;
const detectBtn = document.getElementById('detectBtn') as HTMLButtonElement;
const inferenceTime = document.getElementById('inferenceTime') as HTMLSpanElement;
const thumbnailGrid = document.getElementById('thumbnail-grid') as HTMLDivElement;
const webgpuLabel = document.getElementById('webgpu-label') as HTMLLabelElement;
const epRadios = document.querySelectorAll<HTMLInputElement>('input[name="ep"]');

thresholdSlider.addEventListener('input', () => {
  thresholdLabel.textContent = parseFloat(thresholdSlider.value).toFixed(2);
});

// --- Class names ---
const classNames = await fetch(`${import.meta.env.BASE_URL}classes.txt`)
  .then(r => r.text())
  .then(text => text.trim().split('\n'));

// --- Shared state ---
let currentImage: HTMLImageElement | null = null;

function setCurrentImage(img: HTMLImageElement) {
  currentImage = img;
  const MAX = 800;
  const scale = Math.min(MAX / img.naturalWidth, MAX / img.naturalHeight, 1);
  const dispW = Math.round(img.naturalWidth * scale);
  const dispH = Math.round(img.naturalHeight * scale);
  canvas.width = dispW;
  canvas.height = dispH;
  ctx.drawImage(img, 0, 0, dispW, dispH);
  controls.style.display = 'block';
  canvas.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// --- File input ---
const fileInput = document.getElementById('fileInput') as HTMLInputElement;
fileInput.addEventListener('change', (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file) return;
  const img = new Image();
  img.src = URL.createObjectURL(file);
  img.onload = () => setCurrentImage(img);
});

// --- IIIF Manifest ---

type CanvasEntry = { thumbUrl: string; fullUrl: string; label: string };

function detectVersion(manifest: any): 2 | 3 {
  const ctx = [manifest['@context']].flat().join(' ');
  return ctx.includes('presentation/3') ? 3 : 2;
}

function labelString(label: any): string {
  if (!label) return '';
  if (typeof label === 'string') return label;
  // v3: { "en": ["..."], "ja": ["..."] }
  const values = Object.values(label as Record<string, string[]>);
  return values[0]?.[0] ?? '';
}

function serviceId(service: any): string | undefined {
  if (!service) return undefined;
  const s = Array.isArray(service) ? service[0] : service;
  return (s?.['@id'] ?? s?.id) as string | undefined;
}

function parseV2(manifest: any): CanvasEntry[] {
  const canvases = manifest.sequences?.[0]?.canvases ?? [];
  return canvases.flatMap((canvas: any) => {
    const resource = canvas.images?.[0]?.resource;
    if (!resource) return [];
    const svcId = serviceId(resource.service);
    const fullUrl = (resource['@id'] ?? '') as string;
    return [{
      label: labelString(canvas.label),
      fullUrl,
      thumbUrl: svcId ? `${svcId}/full/150,/0/default.jpg` : fullUrl,
    }];
  });
}

function parseV3(manifest: any): CanvasEntry[] {
  const canvases = manifest.items ?? [];
  return canvases.flatMap((canvas: any) => {
    const body = canvas.items?.[0]?.items?.[0]?.body;
    if (!body) return [];
    const svcId = serviceId(body.service);
    const fullUrl = (body.id ?? body['@id'] ?? '') as string;
    return [{
      label: labelString(canvas.label),
      fullUrl,
      thumbUrl: svcId ? `${svcId}/full/150,/0/default.jpg` : fullUrl,
    }];
  });
}

function renderThumbnails(entries: CanvasEntry[]) {
  thumbnailGrid.innerHTML = '';
  if (entries.length === 0) {
    thumbnailGrid.textContent = 'No canvases found.';
    return;
  }
  for (const entry of entries) {
    const item = document.createElement('div');
    item.className = 'thumb-item';
    item.title = entry.label;

    const thumbImg = document.createElement('img');
    thumbImg.crossOrigin = 'anonymous';
    thumbImg.src = entry.thumbUrl;
    thumbImg.alt = entry.label;
    item.appendChild(thumbImg);

    item.addEventListener('click', () => {
      thumbnailGrid.querySelectorAll('.thumb-item').forEach(el => el.classList.remove('selected'));
      item.classList.add('selected');

      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.src = entry.fullUrl;
      img.onload = () => setCurrentImage(img);
      img.onerror = () => console.error('Image load failed:', entry.fullUrl);
    });

    thumbnailGrid.appendChild(item);
  }
}

const loadManifestBtn = document.getElementById('loadManifestBtn') as HTMLButtonElement;
const manifestUrlInput = document.getElementById('manifestUrl') as HTMLInputElement;

loadManifestBtn.addEventListener('click', async () => {
  const url = manifestUrlInput.value.trim();
  if (!url) return;

  loadManifestBtn.disabled = true;
  loadManifestBtn.textContent = 'Loading...';
  thumbnailGrid.innerHTML = '';

  try {
    const manifest = await fetch(url).then(r => r.json());
    const version = detectVersion(manifest);
    const entries = version === 3 ? parseV3(manifest) : parseV2(manifest);
    console.log(`IIIF v${version}: ${entries.length} canvases`);
    renderThumbnails(entries);
  } catch (err) {
    thumbnailGrid.textContent = `Error: ${err}`;
  } finally {
    loadManifestBtn.disabled = false;
    loadManifestBtn.textContent = 'Load';
  }
});

// --- Detection ---
detectBtn.addEventListener('click', async () => {
  if (!currentImage) return;

  const img = currentImage;
  const threshold = parseFloat(thresholdSlider.value);

  const MAX = 800;
  const scale = Math.min(MAX / img.naturalWidth, MAX / img.naturalHeight, 1);
  const dispW = Math.round(img.naturalWidth * scale);
  const dispH = Math.round(img.naturalHeight * scale);

  ctx.drawImage(img, 0, 0, dispW, dispH);

  predict_canvas.width = INPUT_SIZE;
  predict_canvas.height = INPUT_SIZE;
  predict_ctx.drawImage(img, 0, 0, INPUT_SIZE, INPUT_SIZE);
  const imageData = predict_ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
  const { data } = imageData;

  const float32 = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
  const pixels = INPUT_SIZE * INPUT_SIZE;
  for (let i = 0; i < pixels; i++) {
    float32[i]              = data[i * 4]     / 255;
    float32[i + pixels]     = data[i * 4 + 1] / 255;
    float32[i + pixels * 2] = data[i * 4 + 2] / 255;
  }

  detectBtn.disabled = true;
  detectBtn.textContent = 'Detecting...';

  const inputTensor = new ort.Tensor('float32', float32, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  const start = performance.now();
  const results = await session.run({ images: inputTensor });
  const elapsed = (performance.now() - start).toFixed(1);
  console.log(`推論時間: ${elapsed}ms`);
  inferenceTime.textContent = `${elapsed} ms`;

  detectBtn.disabled = false;
  detectBtn.textContent = 'Start Detection';

  const output = results['output0'].data as Float32Array;
  const scaleX = dispW / INPUT_SIZE;
  const scaleY = dispH / INPUT_SIZE;

  let count = 0;
  for (let i = 0; i < 300; i++) {
    const base = i * 6;
    const score = output[base + 4];
    if (score < threshold) continue;

    const x1 = output[base]     * scaleX;
    const y1 = output[base + 1] * scaleY;
    const x2 = output[base + 2] * scaleX;
    const y2 = output[base + 3] * scaleY;
    const classId = Math.round(output[base + 5]);

    const label = `${classNames[classId] ?? classId} ${(score * 100).toFixed(1)}%`;
    console.log(label);

    ctx.strokeStyle = 'red';
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.fillStyle = 'red';
    ctx.font = '14px sans-serif';
    ctx.fillText(label, x1, y1 > 16 ? y1 - 4 : y1 + 16);
    count++;
  }
  console.log(`Detected: ${count}`);
});

// --- ONNX Runtime WASM paths (same-origin for COEP compatibility) ---
ort.env.wasm.wasmPaths = import.meta.env.BASE_URL;

// --- WebGPU check ---
const webgpuSupported = 'gpu' in navigator;
if (!webgpuSupported) {
  const webgpuRadio = webgpuLabel.querySelector('input')!;
  webgpuRadio.disabled = true;
  webgpuLabel.title = 'WebGPU is not supported in this browser';
  webgpuLabel.style.opacity = '0.4';
}

// --- Session management ---
let session = await ort.InferenceSession.create(`${import.meta.env.BASE_URL}yolo26m.onnx`);
console.log('Model loaded (CPU).');

async function reloadSession(ep: 'cpu' | 'webgpu') {
  detectBtn.disabled = true;
  detectBtn.textContent = 'Loading model...';
  inferenceTime.textContent = '';
  try {
    session = await ort.InferenceSession.create(`${import.meta.env.BASE_URL}yolo26m.onnx`, {
      executionProviders: [ep],
    });
    console.log(`Model loaded (${ep}).`);
  } catch (e) {
    console.error(e);
    alert(`Failed to load model with ${ep}. Falling back to CPU.`);
    const cpuRadio = document.querySelector<HTMLInputElement>('input[value="cpu"]')!;
    cpuRadio.checked = true;
    session = await ort.InferenceSession.create('/yolo26m.onnx');
  } finally {
    detectBtn.disabled = false;
    detectBtn.textContent = 'Start Detection';
  }
}

epRadios.forEach(radio => {
  radio.addEventListener('change', () => {
    if (radio.checked) reloadSession(radio.value as 'cpu' | 'webgpu');
  });
});
