# 次回やること

## IIIFマニフェスト対応

使うマニフェスト: https://dl.ndl.go.jp/api/iiif/3459985/manifest.json (v2)

### 実装の流れ
1. HTMLにマニフェストURL入力欄を追加（file inputの代わり or 併用）
2. マニフェストJSONを fetch で取得
3. v2形式で画像URL一覧を取り出す
   - `manifest.sequences[0].canvases[n].images[0].resource['@id']`
4. サムネイル一覧を表示
5. 画像を選択 → canvasに表示 → YOLO推論 → ボックス描画
   - `img.crossOrigin = 'anonymous'` を忘れずに

### 将来やること
- v3マニフェスト対応
- v2/v3の自動判別
