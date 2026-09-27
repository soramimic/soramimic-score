# Audition contract

Use this checklist for Soramimic Score transcription runs, J-POP comparisons, and private listening reviews. Apply only the items relevant to the requested output.

## Before running

1. Identify each song or excerpt and the requested variant: automatic transcription or supplied lyrics. Record the source audio and exact excerpt bounds.
2. Check the source project's rights and evaluation split. If the material comes from `wav-to-xf-midi`, follow that repository's `AGENTS.md` and current split manifest before opening evaluation references or selecting examples.
3. Inventory the material needed for the chosen variant. An XF MIDI or published lyric source is needed only when the task calls for reference evaluation or supplied lyrics; do not substitute one silently.
4. Hash the source, verified matching vocal stem, and other inference inputs. Record versions and settings for Soramimic Score, separation, and any external comparator.
5. For supplied lyrics, audit full surface text and reading against a reliable complete source. Distinguish lead lyrics from overlapping backing responses and spoken cues. Store the audit privately without copying full copyrighted lyrics into Git.
6. For a multi-song corpus, retain its recorded easy, medium, or hard difficulty labels and the evidence behind them. Give a reason when adding or changing a label.

## Running and reuse

- Keep reference notes, reference timing, and any gold-based render outside the inference path. State explicitly when a tool receives supplied lyrics.
- For a mix/stem A/B test, crop both from the same source interval and match sample rate and tool settings. Do not reuse an unrelated stem merely because the song title matches.
- Soramimic Score may use separated vocals for lyric alignment and original audio for melody estimation. Check the configuration and actual code path when interpreting an A/B result.
- If a source, model, configuration, correction, or intermediate changes, regenerate all affected downstream Score JSON, MIDI, renders, clips, metrics, and review data. Record any safe reuse and its matching provenance.
- Treat an external tool's output as its own baseline. Document its input signal, lyric mode, version, options, and export format. Normalize excerpt offsets and pitch conventions before comparing it with Score or reference notes.

## Review and verification

- Keep a private receipt with completed and failed songs, hashes, variant, versions, reuse decisions, output paths, and any reference used only for display or scoring.
- Verify that Score JSON and exported MIDI decode, note counts and timings agree with the actual outputs, and audio previews play for the intended interval. Do not infer correctness from a file's existence or from note count alone.
- Make source audio, separated vocal, inferred MIDI, and a playable rendering available when they help the requested comparison. Show phrase, mora, F0, candidate, final-note, and reference layers only when that evidence exists; label reference overlays as display-only.
- Keep a multi-song comparison navigable in one review when practical. If the user requests commentable phrase review, verify that comments and ratings persist before sharing it.
- For remote review, follow the private media skill and repository preview rules: use a dedicated allowlisted directory, confirm the exact Tailscale URL and media byte ranges, and reject LAN or public clients. Keep protected source collections, complete lyrics, purchased MIDI, models, and unrelated files outside the served directory.
- Report what changed between conditions, checks actually performed, remaining uncertainty, and the working review URL. Separate an observed output difference from a measured accuracy improvement.
