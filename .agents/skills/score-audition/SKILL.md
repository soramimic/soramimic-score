---
name: score-audition
description: Prepare, compare, and privately review Soramimic Score singing-audio transcriptions, including J-POP mix-versus-vocal tests, automatic or supplied lyrics, and external MIDI baselines. Use for evaluation and listening reviews, not ordinary code edits.
---

# Soramimic Score audition workflow

Use this skill when a request needs reviewable output from a singing-audio transcription. Read the repository's `AGENTS.md` and the relevant CLI or pipeline documentation before running it. Follow the user's requested song, excerpt, tool, and review scope.

## Establish the inputs

- Record whether lyrics come from **automatic transcription** or **supplied lyrics**. Never present one variant as the other. If supplied lyrics are used, audit their completeness and readings; XF lyric events alone do not establish a complete song text.
- Keep the audio source, separated vocal, supplied lyrics, reference score or MIDI, model files, and generated media in private ignored storage. Do not commit them or publish a public review.
- When using another project's corpus, read its data and evaluation rules. In particular, respect the `wav-to-xf-midi` song-family split before selecting J-POP material. Keep reserved test families out of development comparisons.
- Reference notes, XF timing, and known lyrics may be used for an authorized evaluation or as explicitly requested inputs. Never let reference notes or timing enter audio-only inference, and never silently give known lyrics to an automatic-transcription run.

## Compare like with like

- For an original-mix versus separated-vocal test, use the same song, excerpt bounds, sample rate, and conversion settings. Record the source and stem hashes, separation method, tool versions, options, and whether the stem was generated now or reused from verified matching input.
- Inspect the actual signal path before claiming what separation changed. Soramimic Score's standard pipeline separates vocals for reading and alignment while its melody inference can use the original audio. An external Voice-to-MIDI tool may instead receive the separated stem directly.
- Reuse a prior stage only when its inputs, code, model, and settings still match. Rerun every downstream output affected by a changed dependency. A presentation-only review edit does not require new inference.
- Preserve pronunciation alternatives: do not discard a generated reading solely because its mora count differs from another candidate.
- Align exported MIDI and audio to the same excerpt origin before comparing note timing. Report note counts or pitch distributions as observations; do not call them accuracy without a suitable reference and a stated matching rule.

## Deliver a reviewable result

- Preserve a private receipt identifying the lyric variant, input hashes, excerpt, models and settings, fresh or reused stages, output paths, and failures. Keep evaluation truth separate from inference inputs.
- Provide the source or stem audio, the inferred notes and lyrics, and a playable or downloadable output where available. For detailed alignment review, show the evidence the pipeline actually produced, such as phrase bounds, mora intervals, source-vocal F0, candidate notes, final notes, and optional clearly labeled reference overlays. Do not fabricate unavailable layers.
- Use the private Tailscale preview workflow for requested listening or remote review. Serve only explicitly selected review files from a dedicated directory; verify the exact page and media URLs, byte-range playback, and rejection of non-Tailscale clients. Keep the preview available while review is pending.

For the preparation, validation, and handoff checklist, read [references/audition-contract.md](references/audition-contract.md).
