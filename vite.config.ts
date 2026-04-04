import { defineConfig } from 'vite';

export default defineConfig({
  base: '/test_ts_onnx/',
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  build: {
    rollupOptions: {
      input: {
        main: 'index.html',
        realtime: 'realtime_vid.html',
      },
    },
  },
});
