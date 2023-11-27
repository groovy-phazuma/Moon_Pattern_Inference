# 実装・ベンチマーク対象とする手法の調査

### Architectureによる認識性の違い
参考: https://qiita.com/wakayama_90b/items/e2c9f5e65aec08ffc122  
* ヒトの認識は形状（シルエット）に重きを置いている
* CNNはテクスチャを重視する傾向にある。
* ViTはシルエットを重視する傾向にある。

### モデル解釈architectures
参考: https://ai-scholar.tech/articles/explainable.ai/group-cam1
* 領域ベース（RISE, XRAI, Score-CAMなど）
計算コストが高い  

* 活性化ベース（GradCamなど）
ノイズが多く含まれる  
そもそも正しいのか疑問を投げかける論文も存在

* Attention (Transformer系)
Attentionの重みを用いる  
そもそも対応するかは不明  

### 対象としたいモデル
* ViT (transformer baseline)
* Swin Transformer (ViTの改善版)
* ResNet (CNN baseline)
* ConvNext (ResNetの改善版)

### 実装したい判断根拠解析手法
* Attention (transformer系)
* GradCam
* RISE

#### Note
* CNNよりもViTの方が人間の直観に近い結果になるのではないか。(本命)
* 学習時の画像変換や、月の画像をどのようにインプットするかによっても変化しそう。
* 今回、解析対象画像が多くない（月の画像のみ）なので、時間がかかるarchitectureも使用可能か