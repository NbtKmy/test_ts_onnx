# 開発メモ

## onnxruntime-web + Vite dev サーバーの設定

### COEP / COOP ヘッダー

本番環境（GitHub Pages）では `coi-serviceworker` がヘッダーを補完するが、Vite dev サーバーではネイティブに設定が必要。

```ts
// vite.config.ts
server: {
  headers: {
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Cross-Origin-Embedder-Policy': 'require-corp',
  },
},
```

### `public/` の `.mjs` は動的インポートできない

`ort.env.wasm.wasmPaths` に `BASE_URL` を設定すると、ONNX Runtime が `public/` 内の `.mjs` ファイルを動的インポートしようとする。Vite は `public/` のファイルをモジュールとして提供できないため 500 エラーになる。

対策: `wasmPaths` は本番のみ設定し、dev では node_modules から自動解決させる。

```ts
if (import.meta.env.PROD) {
  ort.env.wasm.wasmPaths = import.meta.env.BASE_URL;
}
```

あわせて `optimizeDeps.exclude: ['onnxruntime-web']` も必要。

---

## リアルタイム推論ループの設計

### async rAF + WebGPU で描画が消える問題

`async function` を `requestAnimationFrame` のコールバックに使うと、最初の `await` の時点でブラウザは rAF コールバック完了と判断してペイントする。WebGPU 推論は本物の非同期（GPU への投げっぱなし）なので、推論後に描いたバウンディングボックスが次の rAF の `drawImage` で上書きされてからペイントされ、箱が見えなくなる。

CPU (WASM) では推論が同期的にメインスレッドをブロックするため、`await` がほぼ即時解決し偶然見えていた。

**解決策**: 推論ループと描画ループを分離する。

```ts
// 推論ループ: async while で推論結果をキャッシュに書き込む
async function inferenceLoop() {
  while (running) {
    // ... 推論 ...
    latestDetections = await results['output0'].getData();
    await new Promise(r => setTimeout(r, 0)); // メインスレッドを解放
  }
}

// 描画ループ: 同期 rAF でキャッシュを使って描画（await なし）
function renderLoop() {
  ctx.drawImage(video, ...);
  if (latestDetections) { /* バウンディングボックスを描く */ }
  requestAnimationFrame(renderLoop);
}
```

### CPU (WASM) でブラウザが固まる問題

`while` ループで `await session.run()` が同期的に解決すると、ループがマイクロタスクとして連続実行され event loop が飢餓状態になる。

対策: 各ループ末尾に `await new Promise(r => setTimeout(r, 0))` を置き、マクロタスクとして yield する。

### WebGPU で tensor データを読む

WebGPU バックエンドでは出力テンソルが GPU 上にあるため、`.data` では読めない（空になる）。`getData()` で CPU に転送する。

```ts
// NG (WebGPU では空になる)
const output = results['output0'].data as Float32Array;

// OK (CPU・WebGPU 両対応)
const output = await results['output0'].getData() as Float32Array;
```
