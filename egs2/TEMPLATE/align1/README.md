# 音素CTC強制アライメントの共通レシピ

学習済みの音素CTC ASRモデルを別途用意し、推論と境界評価を行うESPnet2レシピです。
モデル学習とWFSTは対象外です。label-prior CTCで学習したモデルでは、
強制アライメントのViterbi探索に学習済みのlabel priorsを適用します。

## レシピの作成

リポジトリのルートから実行します。

```bash
bash egs2/TEMPLATE/align1/setup.sh egs2/my_corpus/align1
cd egs2/my_corpus/align1
```

`run.sh`、`README.md`、`local/path.sh` はコピーされ、実験条件や環境設定を個別に変更できます。
`align.sh` とユーティリティはTEMPLATEへのシンボリックリンクで共有されます。
既存ファイルを上書きするセットアップは拒否します。データやモデルはコピーしません。

## 入力

デフォルトでは同じコーパスの `../asr1/data/test` を参照します。
`run.sh` の `test_sets` を変更するか、`--test-sets` と `--data-root` で指定してください。

```text
data/eval/
  wav.scp    # 発話IDと音声パス（または信頼できる音声パイプ）
  text       # 発話IDと通常の書き起こし。事前の音素化は不要
  segments   # 任意。録音から発話を切り出す場合
```

推論時の発音（音素列）は、Stage 0でTextGridの音素ラベルを単語ごとにまとめて作成します。
参照の**時刻**は使いませんが、参照と同じ**発音**を与えた条件での境界精度を測ることになります。
TextGridがない発話は推論・評価の対象から除外します（coverage JSONに記録されます）。

音素トークンのモデル（`token_type=phn`、または音素列を `token_type=word` として学習したもの）が対象です。
BPE・文字モデルは受け付けません。
TextGrid変換には `espnet[recipe]` の `praatio` が必要です。

## ステージ

| Stage | 処理 | 必要な入力 |
|---|---|---|
| 0 | TextGridを参照JSONL・発音JSONLに変換 | 書き起こし、TextGrid |
| 1 | 音素・単語区間を推定 | 音声、書き起こし、音素モデル、Stage 0の発音JSONL |
| 2 | 保存した区間を評価 | 参照JSONL、予測JSONL |

デフォルトはStage 1から2です。通常の推論では参照を再生成しません。
各ステージ内で全subsetを処理し、完了してから次のステージに進みます。

```bash
# 参照変換だけ（モデル不要）:
bash run.sh --stage 0 --stop-stage 0 \
    --test-sets eval --data-root data \
    --textgrid-root /path/to/eval_textgrids

# 推論と評価（Stage 0で作成した発音を使う）:
bash run.sh --test-sets eval --data-root data \
    --asr-train-config /path/to/phone_model/config.yaml \
    --asr-model-file /path/to/phone_model/checkpoint.pth

# 保存済みの予測を再評価（モデル・GPU不要）:
bash run.sh --stage 2 --test-sets eval
```

変換から推論・評価まで実行する場合は `--stage 0` にTextGrid・モデルの両方を指定します。
Stage 0の `--textgrid-root` は指定したディレクトリを再帰検索します。
コーパス固有のディレクトリ名への変換は行いません。
ファイル名（拡張子を除く）を発話IDとし、同じIDのTextGridが複数ある場合はエラーにします。
分割ごとにIDが重複するデータでは、それぞれの分割のTextGridディレクトリを指定して別々に実行してください。
`--word-tier`・`--phone-tier` で層名を指定できます（デフォルト：`words`・`phones`）。
IDとファイル名が異なる場合は、`pyscripts/utils/prepare_alignment_reference.py` を
`--textgrid-scp` 付きで直接実行してください。
発話開始を0秒とするTextGridのみが対象です。参照時刻は推論・発音生成には使いません。

Stage 0は書き起こしの各単語をTextGridの単語層に対応付け、
その単語区間内の音素ラベル（空・`sil`・`sp` を除く）を発音とします。
TextGridの単語は大文字・小文字を区別せずに照合し、`<unk>` は任意の単語に一致します。
MFAがアポストロフィで分割した単語（例：`QUILTER'S` → `quilter`・`s`）は1単語にまとめます。
対応付けできない発話があればエラーにします。
`${reference_root}/${dset}.pronunciations.jsonl` と、変換できた発話IDの一覧 `${dset}.keys` を出力し、
Stage 1はこの一覧の発話だけを推論します。

参照の出力先は `--reference-root`（デフォルト：`data/reference`）です。
予測・評価の出力先は `--output-root`（デフォルト：`exp/alignment`）です。
参照の不足・余剰IDはcoverage JSON、予測の欠落や対応区間数は評価結果に記録します。

## 音声形式と追加オプション

デフォルトはCPU・1ワーカーです。`--device cuda:0`・`--nj` で変更できます。
各ワーカーがモデルを読み込むため、並列数を増やす際はメモリ使用量に注意してください。
`--cmd` でESPnetのジョブランナーを指定できます。

`--audio-type kaldi_ark` で波形アーカイブを読み込めます。
サンプリング周波数が保存されていないアーカイブは `--input-fs` も必要です。
`segments` がデータディレクトリにある場合は自動的に切り出しに使います。
出力は発話相対時刻で、切り出し位置も保持します。
音声はモデルのサンプリング周波数へリサンプリングされます。
VAD・時間伸縮・区間長拡張は行いません。

発音JSONLの形式は次のとおりです。
独自の発音を使う場合は、この形式のファイルと発話ID一覧を
`${reference_root}/${dset}.pronunciations.jsonl`・`${dset}.keys` に置いてStage 1から実行してください。

```json
{"utt_id":"u1","words":["THE","CAT"],"phones":[["DH","AH0"],["K","AE1","T"]]}
```

`words` はクリーニング後の書き起こしと一致させます。
`--phone-map` は評価時の音素ラベル変換表（JSON）で、参照・予測の両方に適用します。
モデルは設定・チェックポイントの組か、`--model-tag` のどちらかで指定できます。

## 評価指標

PBE/WBEは各対応区間の開始・終了時刻の絶対誤差を平均し、発話内で平均してから発話間で平均します。
共有境界の重複は除去しません。
PDUR/WDURは同じ対応区間の集合について全発話をまとめた平均区間長で、参照・予測を別々に出力します。
単位はmsです。区間長は誤差ではありません。

対応付けは時刻ではなくラベル列のLevenshteinアライメントで行います。
単語は大文字・小文字を区別せず、音素は完全一致です（明示的な音素変換表を指定した場合を除く）。
空ラベル・長さ0・無音・未知ラベルは除外しますが、時間軸の空白は詰めません。
対応付けできた区間の割合も報告してください。定義できない値は `null`・`NA` になります。

各分割の `score/` に `summary.json`、`summary.tsv`、`per_utterance.jsonl` を保存します。
[LibriSpeechのレシピ](../../librispeech/align1/README.md) に指標・参照データの詳細があります。
LibriSpeechの `run.sh` は、分割名 `test_clean` などをTextGridの `test-clean` などに変換します。
MFAなどで自動生成した参照に対する結果は、人手正解に対する精度ではなく参照との一致度として扱ってください。
