# 開発者向けガイド

音声解析、歌詞と音符の対応付け、JSONの保存・読込をPythonから利用できます。
インストールと音源解析コマンドは[README](../README.md)を参照してください。

## 標準の音声解析

`analyze_audio(path, model_config=config)`は次の処理を行います。

1. Demucs（HTDemucs）で原音からボーカルを分離します。
2. Whisperで原音から歌詞の文と行ごとの時刻を認識します。
3. `soramimic-yomi`で読み候補を生成し、MeCabとUniDic Liteで別の読みを補います。
4. KanaWhisperで原音とボーカルの発音を認識し、読み候補と比較します。
5. ボーカルのReazonSpeechかなCTC出力を使い、選んだ読みに沿って発音単位の時刻を推定します。
6. SheetSage2で原音からメロディの音高と開始・終了時刻を推定します。
7. 発音と音符を対応付け、`ScoreDocument`を返します。

正式歌詞を`lyrics=("一行目", "二行目")`として渡した場合も、Whisperで音源内の位置を探します。
括弧内のかなは、本文の読み・括弧内の読み・両方を歌う読みとして比較します。
通常の認識で区別できなければKanaWhisperと照合し、不確かな括弧は残します。
歌詞本文と元の入力を保持し、読みと時刻を別に推定します。認識結果と入力歌詞を
対応付けたまとまりで読みを選び、CTCで発音時刻を推定します。
日本語向けの解析です。英語や数字もカナ読みに変換しますが、
読めない語やCTCで扱えない文字はエラーになります。

`soramimic-yomi`のユーザー辞書・英語変換・数字の読みを使い、最大32候補を受け取ります。
その第一候補を基本の読みとし、UniDicの最大32通りの形態素解析から、UniDicの第一解析と
同じ単語区切りを持つ異なる読みを追加します。同じカナは一つにまとめ、両方の出典を残します。
UniDicで読めない英語などは`soramimic-yomi`の候補だけを使います。どちらでも読めない場合や
読み生成エンジン自体が失敗した場合はエラーになります。

候補をモーラ数で除外することはありません。候補が複数ある行だけ音声を比較し、
24秒以内の窓で行末まで認識します。読みと音声の発音単位の編集距離を比較し、原音・
ボーカルの一方で悪化する候補や同点の候補は採用しません。認識できなかった場合も
最初の読みを維持します。モデル実行自体の失敗はエラーとして報告します。

`ModelConfig(separate_vocals=False, ...)`で分離を、`acoustic_readings=False`で音声比較を
それぞれ省略できます。分離を省略すると読み選択・発音時刻推定には原音だけを使います。
音声比較を省略しても、読み生成には`soramimic-yomi`とUniDicを使います。
Demucsの重みは公式HTDemucsチェックポイントを使い、取得済みファイルは
`demucs_checkpoint=Path(...)`でも指定できます。KanaWhisperを変更する場合は`kana_model`に
モデルIDまたはローカルディレクトリを指定します。`local_files_only=True`はこれらの取得も禁止します。

音声モデルは呼び出す段階で読み込み、推論後に解放します。CPUとCUDAを選択できます。
SheetSage2は指定ディレクトリの付属コードを実行します。重み・コードとも信頼できる
配布元から用意してください。モデルの利用条件は[外部ライブラリ・モデルの表記](../THIRD_PARTY_NOTICES.md)
を参照してください。

## モデルの差し替え

独自の推論処理を使う場合は、4つの関数を`AudioAdapters`に指定します。

```python
from soramimic_score import AudioAdapters, analyze_audio, dump

adapters = AudioAdapters(
    lyric_recognizer=recognize_lyrics,
    reading_selector=select_readings,
    mora_aligner=align_moras,
    melody_transcriber=transcribe_melody,
)
score = analyze_audio("song.wav", adapters)
dump(score, "work/song.score.json")
```

上記の関数は利用側で定義します。モデル設定と`adapters`の同時指定はできません。

| 関数 | 入力 | 戻り値の各要素 |
| --- | --- | --- |
| 歌詞認識 | 音源の`Path` | `LyricLine`：文字列、開始・終了時刻 |
| 読み選択 | 音源、歌詞行 | `ReadingSelection`：カナ、出典、スコア、任意の候補・判断根拠 |
| 発音時刻推定 | 音源、歌詞行、読み | `AlignedMora`：行番号、発音単位番号、カナ、時刻、信頼度 |
| メロディ推定 | 音源 | `MelodyNote`：時刻、MIDI音高、出典、任意の信頼度 |

時刻は入力音源の先頭からの秒数です。歌詞行、発音、メロディはそれぞれ時系列順で、
同種の区間が重ならないように渡します。発音列は選択した読みのモーラ列と一致させます。
音高の信頼度が不明なら`None`を使います。辞書の読みスコアやCTCスコアは、
認識の正しさを保証する確率ではありません。
`ReadingSelection.candidates`には比較したカナ候補、`detail`にはJSON化できる判断根拠を
渡せます。標準構成では候補ごとの生成元（`candidate_provenance`）と、音声の認識結果・
比較距離・選択理由・音声区間を保持します。`soramimic-yomi`の候補順位・生成規則も残します。
これらは出力JSONの`observations.evidence`に`kind: "reading-selection"`として保存され、
選択済みの読みから参照できます。読みのスコア`1.0`は選択を固定するための値であり、
音響的な正しさの確率ではありません（根拠中の`confidence_available`は`false`）。

## 観測データからの変換

`build_audio_observations(lines, readings, aligned_moras, melody_notes)`はモデルの結果を
`IntermediateRepresentation`へまとめます。これは歌詞・読み・発音時刻・音符候補・
根拠を保持する、バージョン付きの入力形式です。

```python
from soramimic_score import compile_score, dump, load

score = compile_score(observations)
dump(score, "work/song.score.json")
same_score = load("work/song.score.json")
```

`compile_score`には、対応付け前の観測を渡します。対応付け済みの場合は
`from_linked_observations`を使うと、再計算せず歌唱用の音符列を組み立てられます。
`build_known_lyrics_document`は指定歌詞にも自動認識で選択済みの歌詞にも利用できる
補助関数で、それ自体は音声認識を行いません。

## 対応付けの詳細

標準の対応付けは、発音時刻とメロディノートを使って、各音符で何を発音するかを決めます。
同じ発音に属する同音高の断片をまとめ、音高が変わっても発音が続く場合は継続音にします。
対応が見つからない歌詞は削除せず、未解決として保持します。

細かな設定は`soramimic_score.pipeline.run_stage3_document`で指定できます。
歌詞行の時刻には`line_windows_by_utterance`を使います。反復発声の回数を音響根拠から
補う場合は、`vocalization_reattacks_by_utterance`に`VocalizationReattack`を渡します。
標準の音源解析は、この反復補完用の追加認識を実行しません。

比較実験には`align_correspondence`と`run_correspondence_document`も使えます。
これらは発音と音符のまとまりをsemi-Markov方式で探索する別の対応付けアルゴリズムです。
標準の音源解析や`compile_score`では呼び出しません。

## 出力JSON

`ScoreDocument`は元の観測、対応付け、根拠、歌唱用の音符列を一つに保存します。
結合・分割した音符は元の候補を参照する根拠を持ち、保存した結果から追跡できます。

- `canonical`：歌詞と発音単位のID
- `performed`：観測状態と対応付け
- `synthesis_plan`：音高・時刻・発音を持つ歌唱用の音符列

形式名は`format: "soramimic-score"`、形式の版は整数の`schema_version`です。
パッケージのバージョンとは独立しています。

CLIで観測JSONを変換する場合は次のように指定します。

```sh
soramimic-score --input work/observations.json --output work/song.score.json
```

対応付け済みの入力には`--linked-input`を付けます。
