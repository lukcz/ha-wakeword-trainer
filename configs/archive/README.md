# Archived VAD Presets

This directory contains older or more experimental Polish VAD presets that were
kept for reference after the top-level `configs/` directory was simplified.

Use the top-level presets first:

- `configs/polish_vad.yaml`
- `configs/polish_vad_balanced.yaml`
- `configs/polish_vad_high_recall.yaml`
- `configs/polish_vad_low_fp.yaml`
- `configs/polish_vad_public_research.yaml`

Archive groups:

- `polish_vad_public_large*`: older public-bootstrap variants
- `polish_vad_public_research_*`: ablations around the research preset
- `polish_vad_public_hardneg*`: stronger hard-negative experiments
- `polish_vad_public_xlarge.yaml`: scale-only experiment
- `polish_vad_strict.yaml`, `polish_vad_very_low_fp.yaml`, `polish_vad_low_fp_stable.yaml`: extra low-FP variants

Keep these only when you want to reproduce an older run or compare against a
specific archived experiment.
