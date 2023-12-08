# シルエット画像を深層学習でどの程度分類できるか

## Introduction
* 月面画像の分類用のモデルを構築したい
* そのために、シルエット画像の分類モデルを用意する
* どの程度精度が出るかを検証する

## Strategy
1. ImageNet-Sのピクセル単位でのsegmentation Datasetを準備する
1. ViT, ResNet等のモデルでシルエット画像を学習させ、分類タスク精度を評価する (Simple Ver)
1. データセットはImageNetのラベルに基づいている。秋田犬と柴犬を統合して犬のラベルを付けるなど、ラベル統合を行う
1. 再評価を行う (Label Modified Ver)

## Method (Memo)
参照: Moon_Pattern_Inference/silhouette_model/train.py

#### 学習
* いくつかlrをふる
* silhouetteに必要のないaugmentationは実装しない（color jitterなど）
* affine変換は用いる

#### ラベル統合
* ImageNetのラベルはツリー構造になっている（WordNet）
* 

## Result
#### Simple Ver, validation accuracy (Top1)

| Learning Rate | ViT | ResNet50 |
| :--- | :---: | :---: |
| 3e-6 | 0.5370 | 0.2628 |
| 1e-5 | 0.6254 | 0.3173 |
| 3e-5 | 0.5841 | 0.3770 |
| 1e-4 | 0.5660 | 0.4216 |
| 3e-4 | 0.4347 | 0.3260 |
| 1e-3 | 0.5287 | 0.2250 |
| 3e-3 | --- | 0.2978 |

* batch_size=128, epoch=100
* lossの初期の落ち方はそこまで高くなかったので、pretrainedを用いる必要はないかもしれない
* 全体的に精度はそこまで高くない
* ViTの方が精度が高い
* 画像数を増やす意味も含め、クラスラベルをいくつか統一するのがやはり妥当であろう

#### Label Modified Ver, validation accuracy (Top1)

