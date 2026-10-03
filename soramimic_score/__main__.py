"""Compile observation JSON into one canonical Soramimic Score document."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .document import compile_score, dump, from_linked_observations
from .ir import IntermediateRepresentation


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        return analyze_main(sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__,
                                     epilog="音源の解析: soramimic-score analyze --help")
    parser.add_argument("--input", type=Path, required=True, help="Observation JSON")
    parser.add_argument("--output", type=Path, required=True, help="Soramimic Score JSON")
    parser.add_argument(
        "--linked-input",
        action="store_true",
        help="Input already contains correspondence links; compile without decoding",
    )
    args = parser.parse_args()
    document = IntermediateRepresentation.from_json(args.input.read_text(encoding="utf-8"))
    result = (
        from_linked_observations(document)
        if args.linked_input
        else compile_score(document)
    )
    dump(result, args.output)
    print(
        f"{len(result.score.synthesis_plan)} slots; "
        f"{len(result.score.unresolved_unit_ids)} unresolved units"
    )
    return 0


def analyze_main(argv) -> int:
    from .audio import analyze_audio
    from .models import ModelConfig
    parser = argparse.ArgumentParser(description="歌唱音源から歌詞つき音符JSONを生成")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sheetsage-model", type=Path, required=True)
    parser.add_argument("--sheetsage-base", type=Path, required=True)
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--ctc-model", default="reazon-research/japanese-wav2vec2-base-rs35kh")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-vocal-separation", action="store_true", help="ボーカル分離を省略")
    parser.add_argument("--dictionary-readings", action="store_true", help="音声比較を省略し辞書の読みを使用")
    parser.add_argument("--demucs-checkpoint", type=Path, help="取得済みHTDemucsチェックポイント")
    parser.add_argument("--kana-model", default="sbintuitions/kana-whisper")
    parser.add_argument("--romaji-model", type=Path,
                        help="RomajiASR ONNXモデル。歌詞の欠損を推定カタカナで補う")
    parser.add_argument("--lyrics", type=Path, help="UTF-8歌詞ファイル（1行1フレーズ）")
    parser.add_argument("--adjust-lyrics", action="store_true",
                        help="音源の認識結果に合わせ、入力歌詞を行単位で削除・補完する")
    args = parser.parse_args(argv)
    if args.adjust_lyrics and args.lyrics is None:
        parser.error("--adjust-lyrics には --lyrics が必要です")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.audio.resolve() == args.output.resolve() or (
        args.lyrics is not None and args.lyrics.resolve() == args.output.resolve()
    ):
        parser.error("出力先は入力ファイルと別にしてください")
    try:
        lyrics = (tuple(line for line in args.lyrics.read_text(encoding="utf-8").splitlines()
                        if line.strip()) if args.lyrics else None)
        config = ModelConfig(args.sheetsage_model, args.sheetsage_base, args.whisper_model,
                             args.ctc_model, args.device, args.local_files_only,
                             separate_vocals=not args.no_vocal_separation,
                             acoustic_readings=not args.dictionary_readings,
                             demucs_checkpoint=args.demucs_checkpoint, kana_model=args.kana_model,
                             romaji_model=args.romaji_model)
        score = analyze_audio(args.audio, model_config=config, lyrics=lyrics,
                              adjust_lyrics=args.adjust_lyrics,
                              on_progress=lambda stage: print(stage, file=sys.stderr, flush=True))
        dump(score, args.output)
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"解析に失敗しました: {exc}\n")
    print(f"{len(score.score.synthesis_plan)} slots; saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
