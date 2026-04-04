import './style.css';
import * as ort from 'onnxruntime-web';

const INPUT_SIZE = 640;
const MODEL_FILE = 'yolo26s.onnx';

// --- UI ---
document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <section id="center">
    <h1>YOLO26s Realtime Object Detection</h1>

    <p id="about">
      Realtime Object Detection from Webcam —
      Object detection running entirely in your browser via <a href="https://onnxruntime.ai/docs/get-started/with-javascript/web.html" target="_blank" rel="noopener">ONNX Web Runtime</a>.
      Model: <a href="https://docs.ultralytics.com/models/yolo26/" target="_blank" rel="noopener">YOLO26s</a>.
    </p>

    <div id="controls">
      <div id="ep-selector">
        <span>Backend</span>
        <label><input type="radio" name="ep" value="cpu" checked /> CPU</label>
        <label id="webgpu-label"><input type="radio" name="ep" value="webgpu" /> WebGPU</label>
      </div>
      <label id="threshold-label-wrap">
        Threshold: <span id="thresholdLabel">0.25</span>
        <input type="range" id="thresholdSlider" min="0.05" max="0.95" step="0.05" value="0.25" />
      </label>
      <button id="startBtn">Start Camera</button>
      <span id="fpsDisplay"></span>
    </div>

    <canvas id="canvas"></canvas>
    <canvas id="for_prediction" style="display:none;"></canvas>
  </section>
`;

// --- Elements ---
const canvas      = document.getElementById('canvas') as HTMLCanvasElement;
const ctx         = canvas.getContext('2d')!;
const predCanvas  = document.getElementById('for_prediction') as HTMLCanvasElement;
const predCtx     = predCanvas.getContext('2d')!;
const startBtn    = document.getElementById('startBtn') as HTMLButtonElement;
const fpsDisplay  = document.getElementById('fpsDisplay') as HTMLSpanElement;
const threshSlider = document.getElementById('thresholdSlider') as HTMLInputElement;
const threshLabel  = document.getElementById('thresholdLabel') as HTMLSpanElement;
const webgpuLabel  = document.getElementById('webgpu-label') as HTMLLabelElement;
const epRadios     = document.querySelectorAll<HTMLInputElement>('input[name="ep"]');

threshSlider.addEventListener('input', () => {
  threshLabel.textContent = parseFloat(threshSlider.value).toFixed(2);
});

// --- Class names ---
const classNames = await fetch(`${import.meta.env.BASE_URL}classes.txt`)
  .then(r => r.text())
  .then(t => t.trim().split('\n'));

// --- ONNX setup ---
if (import.meta.env.PROD) {
  ort.env.wasm.wasmPaths = import.meta.env.BASE_URL;
}

const webgpuSupported = 'gpu' in navigator;
if (!webgpuSupported) {
  const radio = webgpuLabel.querySelector('input')!;
  radio.disabled = true;
  webgpuLabel.title = 'WebGPU not supported in this browser';
  webgpuLabel.style.opacity = '0.4';
}

let session = await ort.InferenceSession.create(`${import.meta.env.BASE_URL}${MODEL_FILE}`);
console.log('Model loaded (cpu).');

// --- Pre-allocated buffers ---
predCanvas.width  = INPUT_SIZE;
predCanvas.height = INPUT_SIZE;
const float32 = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);

// --- State ---
const video = document.createElement('video');
video.autoplay = true;
video.playsInline = true;
video.muted = true;

let running = false;
let animId  = 0;
let fpsCnt  = 0;
let fpsT0   = 0;

// --- Backend switch ---
async function reloadSession(ep: 'cpu' | 'webgpu') {
  const wasRunning = running;
  if (running) stopCamera();

  startBtn.disabled = true;
  startBtn.textContent = 'Loading model...';
  try {
    session = await ort.InferenceSession.create(
      `${import.meta.env.BASE_URL}${MODEL_FILE}`,
      { executionProviders: [ep] }
    );
    console.log(`Model loaded (${ep}).`);
  } catch (e) {
    console.error(e);
    alert(`Failed to load model with ${ep}.`);
    document.querySelector<HTMLInputElement>('input[value="cpu"]')!.checked = true;
    session = await ort.InferenceSession.create(`${import.meta.env.BASE_URL}${MODEL_FILE}`);
  } finally {
    startBtn.disabled = false;
    startBtn.textContent = 'Start Camera';
    if (wasRunning) await startCamera();
  }
}

epRadios.forEach(r => r.addEventListener('change', () => {
  if (r.checked) reloadSession(r.value as 'cpu' | 'webgpu');
}));

// --- Camera ---
async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  await video.play();

  canvas.width  = video.videoWidth  || 640;
  canvas.height = video.videoHeight || 480;

  running = true;
  latestDetections = null;
  startBtn.textContent = 'Stop Camera';
  fpsCnt = 0;
  fpsT0  = performance.now();

  inferenceLoop();              // async: 推論を別途回す
  animId = requestAnimationFrame(renderLoop); // 同期 rAF で描画
}

function stopCamera() {
  running = false;
  cancelAnimationFrame(animId);
  (video.srcObject as MediaStream | null)?.getTracks().forEach(t => t.stop());
  video.srcObject = null;
  startBtn.textContent = 'Start Camera';
  fpsDisplay.textContent = '';
}

// --- 推論結果キャッシュ ---
let latestDetections: Float32Array | null = null;
let latestSx = 1;
let latestSy = 1;

// 推論ループ: async while で推論だけ回す
async function inferenceLoop() {
  const px = INPUT_SIZE * INPUT_SIZE;
  while (running) {
    predCtx.drawImage(video, 0, 0, INPUT_SIZE, INPUT_SIZE);
    const { data } = predCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
    for (let i = 0; i < px; i++) {
      float32[i]        = data[i * 4]     / 255;
      float32[i + px]   = data[i * 4 + 1] / 255;
      float32[i + px*2] = data[i * 4 + 2] / 255;
    }
    const tensor  = new ort.Tensor('float32', float32, [1, 3, INPUT_SIZE, INPUT_SIZE]);
    const results = await session.run({ images: tensor });
    if (!running) break;
    latestDetections = await results['output0'].getData() as Float32Array;
    latestSx = canvas.width  / INPUT_SIZE;
    latestSy = canvas.height / INPUT_SIZE;

    // FPS カウント
    fpsCnt++;
    const now = performance.now();
    if (now - fpsT0 >= 500) {
      const fps = fpsCnt / (now - fpsT0) * 1000;
      fpsDisplay.textContent = `${fps.toFixed(1)} fps`;
      fpsCnt = 0;
      fpsT0  = now;
    }

    // CPU(WASM) はメインスレッドを同期的にブロックするため、
    // マクロタスクとして yield して rAF など他のタスクに制御を返す
    await new Promise(r => setTimeout(r, 0));
  }
}

// 描画ループ: 同期 rAF で映像 + 最新検出結果を描く（await なし）
function renderLoop() {
  if (!running) return;

  // canvas サイズを映像に合わせる
  if (video.videoWidth && (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight)) {
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
  }

  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  if (latestDetections) {
    const threshold = parseFloat(threshSlider.value);
    ctx.font = '13px monospace';
    for (let i = 0; i < 300; i++) {
      const b = i * 6;
      const score = latestDetections[b + 4];
      if (score < threshold) continue;

      const x1 = latestDetections[b]     * latestSx;
      const y1 = latestDetections[b + 1] * latestSy;
      const x2 = latestDetections[b + 2] * latestSx;
      const y2 = latestDetections[b + 3] * latestSy;
      const cls = Math.round(latestDetections[b + 5]);
      const lbl = `${classNames[cls] ?? cls} ${(score * 100).toFixed(0)}%`;

      const tw = ctx.measureText(lbl).width + 8;
      const ty = y1 > 18 ? y1 - 18 : y1;

      ctx.strokeStyle = '#00ff44';
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      ctx.fillStyle = 'rgba(0,180,50,0.75)';
      ctx.fillRect(x1, ty, tw, 18);
      ctx.fillStyle = '#fff';
      ctx.fillText(lbl, x1 + 4, ty + 13);
    }
  }

  animId = requestAnimationFrame(renderLoop);
}

// --- Start/Stop button ---
startBtn.addEventListener('click', async () => {
  if (running) {
    stopCamera();
    return;
  }
  startBtn.disabled = true;
  try {
    await startCamera();
  } catch (e) {
    console.error(e);
    alert('カメラへのアクセスに失敗しました。');
  } finally {
    startBtn.disabled = false;
  }
});
