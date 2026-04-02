import * as ort from 'onnxruntime-web';
//import * as ort from 'onnxruntime-web/webgpu';

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <section id="center">
    <h1>Onnx Test with Tiny YOLO3</h1>
    <input type="file" accept="image/*" id="fileInput" />
    <canvas id="canvas"></canvas>
    <canvas id="for_prediction" style="display:none;"></canvas>
  </section>
  `;

/*
Define canvas
*/

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const ctx = canvas.getContext('2d')!;
const predict_canvas = document.getElementById('for_prediction') as HTMLCanvasElement;
const predict_ctx = predict_canvas.getContext('2d')!;

/*
fetching class names
*/
const classNames = await fetch('/classes.txt')
      .then(r => r.text())
      .then(text => text.trim().split('\n'));


/*
file input
*/
const fileInput = document.getElementById('fileInput') as HTMLInputElement;

  fileInput.addEventListener('change', (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file) return;

    const img = new Image();
    img.src = URL.createObjectURL(file);
    img.onload = async () => {

      //const origH = img.naturalHeight;
      //const origW = img.naturalWidth;

      const MAX = 800;
      const scale = Math.min(MAX / img.naturalWidth, MAX / img.naturalHeight, 1);
      const dispW = Math.round(img.naturalWidth * scale);
      const dispH = Math.round(img.naturalHeight * scale);
      
      canvas.width = dispW;
      canvas.height = dispH;
      // ctx.drawImage(img, 0, 0, 416, 416);
      ctx.drawImage(img, 0, 0, dispW, dispH);

      predict_canvas.width = 416;
      predict_canvas.height = 416;
      predict_ctx.drawImage(img, 0, 0, 416, 416);
      const imageData = predict_ctx.getImageData(0, 0, 416, 416);
      const { data } = imageData; // RGBA, 416*416*4個の値

      const float32 = new Float32Array(3 * 416 * 416);
      for (let i = 0; i < 416 * 416; i++) {
        float32[i]             = data[i * 4]     / 255; // R
        float32[i + 416 * 416] = data[i * 4 + 1] / 255; // G
        float32[i + 416 * 416 * 2] = data[i * 4 + 2] / 255; // B
      }

      console.log(float32);
      const inputTensor = new ort.Tensor('float32', float32, [1, 3, 416, 416]);
      //const imageShapeTensor = new ort.Tensor('float32', new Float32Array([416, 416]), [1, 2]);
      const imageShapeTensor = new ort.Tensor('float32', new Float32Array([dispH, dispW]), [1, 2]);
      const start = performance.now();

      const results = await session.run({
        'input_1': inputTensor,
        'image_shape': imageShapeTensor,
      });

      const end = performance.now();
      console.log(`推論時間: ${(end - start).toFixed(1)}ms`);
      

      const boxes = results['yolonms_layer_1'].data as Float32Array;
      const indices = results['yolonms_layer_1:2'].data as Int32Array;
      const scores = results['yolonms_layer_1:1'].data as Float32Array;
      let maxScore = 0;
      for (let j = 0; j < scores.length; j++) {
        if (scores[j] > maxScore) maxScore = scores[j];
      }
      console.log('max score:', maxScore);

      const numDetections = indices.length / 3;
      for (let i = 0; i < numDetections; i++) {
        const boxIdx = indices[i * 3 + 2]; // 3番目の値がボックス番号
        const y1 = boxes[boxIdx * 4];
        const x1 = boxes[boxIdx * 4 + 1];
        const y2 = boxes[boxIdx * 4 + 2];
        const x2 = boxes[boxIdx * 4 + 3];

        const y1c = Math.max(0, y1);
        const x1c = Math.max(0, x1);
        const y2c = Math.min(dispH, y2);
        const x2c = Math.min(dispW, x2);

        ctx.strokeStyle = 'red';
        ctx.lineWidth = 2;
        //ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        ctx.strokeRect(x1c, y1c, x2c - x1c, y2c - y1c);

        //const boxIdx = indices[i * 3 + 2];
        const classIdx = indices[i * 3 + 1];
        const score = scores[classIdx * 2535 + boxIdx];
        console.log(`クラス${classIdx}, スコア: ${score}`);

        const label = `${classNames[classIdx]} ${(score * 100).toFixed(1)}%`;
        ctx.fillStyle = 'red';
        ctx.font = '14px sans-serif';
        // ctx.fillText(label, x1, y1 - 4);
        ctx.fillText(label, x1c, y1c > 16 ? y1c - 4 : y1c + 16);

        
      }
    };
  });

/*
load onnx model
*/

const session = await ort.InferenceSession.create('/tiny-yolov3-11.onnx');
/* const session = await ort.InferenceSession.create('/tiny-yolov3-11.onnx', {
    executionProviders: ['webgpu'],
  });
*/