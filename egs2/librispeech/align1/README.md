# 音素CTCによる強制アライメント

**別途用意した音素CTC ASRモデルのチェックポイント**を使って、独立した推論・評価を行うレシピです。
既存の `asr1` レシピ、データ、モデルは変更しません。
学習、WFST、区間長を拡張する処理のステージは含みません。
`ctc_type: label_prior` で学習したモデルでは、Viterbi探索に学習済みのlabel priorsを適用します。

Stage 0〜2の共通処理は [TEMPLATE/align1](../../TEMPLATE/align1/README.md) の
`align.sh` を共有します。`run.sh` はLibriSpeechの分割名・パスを指定するラッパーです。
ほかのコーパスのレシピは、リポジトリのルートから次のように作成できます。

```bash
bash egs2/TEMPLATE/align1/setup.sh egs2/my_corpus/align1
```

## 評価用の参照アノテーション

既存のLibriSpeech音声と
[ZenodoのLibriSpeech TextGrid](https://zenodo.org/records/2619474) を組み合わせて使います。
これらはMFAで自動生成された参照アノテーションであり、**人手で作成した正解データではありません**。
アーカイブを展開してから、`run.sh` のStage 0で変換してください。

```bash
cd egs2/librispeech/align1
bash run.sh --stage 0 --stop-stage 0 \
    --textgrid-root /path/to/librispeech-align \
    --test-sets "dev_clean dev_other test_clean test_other"
```

`--textgrid-root` には、`test-clean` などのサブディレクトリを含む展開先を指定します。
書き起こしは `--data-root`（デフォルト：`../asr1/data`）から読み、
JSONLとcoverage JSONを `--reference-root`（デフォルト：`data/reference`）に保存します。
推論用の発音JSONL（`<set>.pronunciations.jsonl`）と変換できた発話IDの一覧（`<set>.keys`）も保存します。
Stage 0は指定された参照JSONLを再生成します。モデルや音声の読み込みは不要です。

必要に応じて、既存のレシピ用追加依存パッケージをインストールしてください
（`pip install 'espnet[recipe]'`）。TextGridの変換には `praatio` が必要です。
デフォルトの層（tier）名は `words` と `phones` です。
元の時刻、空ラベル・無音の区間、未知語・未知音素のラベルは保持されます。
不足・余剰のIDは、出力と同じ場所に作成されるcoverage JSONに記録されます。
IDとパスの対応や層名を変更する場合は、
`pyscripts/utils/prepare_alignment_reference.py` を直接実行し、
`--textgrid-scp`、`--word-tier`、`--phone-tier` を指定してください。
受け付けるのは、発話の開始を0秒とする発話相対時刻のTextGridのみです。
録音全体を基準とするアノテーションは、使用前に明示的に変換してください。

## 推論と評価

ステージは **0：参照変換、1：推論、2：評価** です。
デフォルトはStage 1から2なので、通常の推論では参照を再生成しません。
デフォルトの対象は `test_clean` です。複数セットは `--test-sets` で指定してください。

```bash
# 変換済みの参照・発音を使って推論・評価する:
bash run.sh \
    --asr-train-config /path/to/phone_model/config.yaml \
    --asr-model-file /path/to/phone_model/checkpoint.pth

# 参照変換から推論・評価まで実行する:
bash run.sh --stage 0 \
    --textgrid-root /path/to/librispeech-align \
    --asr-train-config /path/to/phone_model/config.yaml \
    --asr-model-file /path/to/phone_model/checkpoint.pth

# モデルやGPUを使わず、保存済みの結果を再評価する:
bash run.sh --stage 2
```

推論の発音には、TextGrid（MFA）の音素ラベルを単語ごとにまとめたものを使います。
参照の時刻は使いませんが、**参照と同じ発音を与えた条件での境界精度**になる点に注意してください。
G2Pで発音を作ると、MFAとの発音の違い（CMUdictの別発音の選択など）による対応付けの失敗が評価に混ざるためです。
TextGridのない発話（test_otherの `8131-117029-0017`）は推論・評価から除外し、coverage JSONに記録します。
TextGridの単語は大文字・小文字を区別せずに照合し、`<unk>` は任意の単語に一致します。
MFAがアポストロフィで分割した単語（例：`QUILTER'S` → `quilter`・`s`）は1単語にまとめます。
音素は空・`sil`・`sp` を除き、`spn` は残します（`../asr1/local/textgrid_to_phoneme.py` と同じ扱い）。

音素ラベルは**チェックポイントの学習時の語彙と一致している必要があります**。
強勢を表す記号・数字は自動では除去しません。
`token_type=phn`、または音素列を `token_type=word` として学習したモデルが対象です。
BPEモデルのチェックポイントは音素モデルとしては使えません。
共通ASRシェルレシピは現時点で `phn` に対応していないため、モデル学習は意図的にこのレシピの対象外としています。

デフォルトはCPU・1ワーカーです。GPUを使う場合は `--device cuda:0` を明示的に指定してください。
`--nj` を増やすとワーカーごとにモデルが読み込まれるため、メモリ使用量に注意してください。
調整はdevセットで行い、test-cleanとtest-otherの結果は分けて報告してください。
使用するチェックポイントの学習・調整に、これらのテストアノテーションを使わないでください。

## 他のデータセットでの利用

`setup.sh` でレシピを作成し、`align.sh` のオプションでデータ・モデルを指定します。
推論だけを行う場合は `espnet2.bin.alignment_inference` を直接実行できます。

```bash
python -m espnet2.bin.alignment_inference \
    --wav-scp data/eval/wav.scp --text data/eval/text \
    --pronunciations pronunciations.jsonl \
    --asr-train-config /path/to/config.yaml \
    --asr-model-file /path/to/checkpoint.pth \
    --output exp/alignment/eval/hyp.jsonl
```

Stage 1は、通常の音声ファイルまたは信頼できるKaldi音声パイプを読み込みます。
波形アーカイブには `--audio-type kaldi_ark` を指定してください。
サンプリング周波数が保存されていないアーカイブでは、`--input-fs` も必要です。
特徴量行列は波形としては受け付けません。
`--segments` を使うと録音から発話区間を切り出します
（`align.sh` ではデータディレクトリに `segments` があれば自動で指定します）。
出力時刻は発話相対時刻になり、実際の切り出し位置もメタデータとして保持されます。
VADによるトリミングや時間伸縮は行いません。
音声はチェックポイントのフロントエンドに設定されたサンプリング周波数へリサンプリングされます。

`--pronunciations` には次の形式のJSONLを指定します（Stage 0が出力するものと同じ形式です）。
一部の発話だけに発音がある場合は、その発話IDの一覧を `--key-file` で指定します。

```json
{"utt_id":"u1","words":["THE","CAT"],"phones":[["DH","AH0"],["K","AE1","T"]]}
```

単語列は、テキストクリーナー適用後の書き起こしの単語列と一致させ、各単語に1つの発音を指定してください。
すべての音素は、モデルの予約トークン以外のトークンである必要があります。
発音の生成に参照アノテーションの時刻は使用しません。

予測と参照アノテーションは、同じJSONL形式を使います。

```json
{"utt_id":"u1","duration_s":1.0,"words":[{"label":"CAT","start":0.1,"end":0.8}],"phones":[{"label":"K","start":0.1,"end":0.2},{"label":"AE1","start":0.3,"end":0.6},{"label":"T","start":0.7,"end":0.8}]}
```

予測された各音素には、所属する単語のインデックスも保存されます。
単語区間の開始・終了は、その単語の最初・最後の音素の境界です。
したがって、単語区間には音素間のCTC blankに対応する空白時間も含まれます。

## 評価指標

[Huangらの論文の3.2節](https://arxiv.org/html/2406.02560v3#S3.SS2) に従い、
PBE（音素境界誤差）・WBE（単語境界誤差）は、対応付けられた各区間について
`(開始時刻の絶対誤差 + 終了時刻の絶対誤差) / 2` を計算します。
まず発話内で平均し、**次に発話ごとの誤差を平均します**。
隣接区間で共有される境界の重複は除去しません。
これは、境界を重複除去してマイクロ平均MAEを求める現在のFA-Benchとは異なります。

PDUR（音素区間長）・WDUR（単語区間長）は、**同じ対応付け済み区間の集合**について、
全発話の区間をまとめて求める平均区間長です。
予測と参照の値を別々に出力します。すべての指標の単位はmsです。
区間長は誤差ではないため、小さいほど良いとは限りません。

単語は大文字・小文字を区別せずに対応付けますが、アポストロフィは保持します。
音素は完全一致で対応付けます。
ラベルを変換する場合は、`align.sh` の `--phone-map` でJSON形式の対応表を指定してください。
この対応表は予測・参照の両方の音素層に適用されます。
対応付けには、ラベルのみを使う決定的なSPLETのLevenshteinアライメントを使用し、
時刻が最も近い区間を選ぶ方法は使いません。
空ラベル・長さ0の区間、無音（`sil`、`sp`、`spn`、`<sil>`、`<eps>`）、
`<unk>`・`<unknown>` は評価対象から除外しますが、時間軸上の空白は詰めません。

各サブセットについて、`score/summary.json`、`summary.tsv`、
`per_utterance.jsonl` を出力します。
対応付けできた・できなかった区間数、除外区間、欠落・空の予測、利用できない層を明示します。
定義できない指標は0ではなく `null`・`NA` として出力します。
発音の違いがある場合に対応付けできた区間だけで評価する処理は、
論文の対応区間に対する式を拡張したものです。
論文の表の厳密な再現とは主張せず、対応付けできた区間の割合も併せて報告してください。
