# Version Layout

This repository keeps two comparable CSCI-V + V-JEPA code paths.

## Current root version: dense-patch V-JEPA

The repository root contains the current local method:

- V-JEPA tokens: 392 dense tokens per 4-frame clip.
- Adapter: projects V-JEPA token width from 1664 to 1024.
- Fusion input: 1 tracklet token + 4 frame tokens + 1024 CSCI-V RGB patch tokens + 392 V-JEPA tokens.
- Total fusion length: 1421 tokens.
- Main config: `configs/mevid_eva02_l_cloth_jepa.yml`.

This version is intended for the next MEVID comparison run.

## Backup version: `legacy_server_compressed_context/`

This folder contains the older server code snapshot:

- V-JEPA tokens are compressed into `MODEL.JEPA_CONTEXT_TOKENS: 4`.
- Fusion input is 1 tracklet token + 4 frame tokens + 4 JEPA context tokens.
- Total fusion length: 9 tokens.
- This is the version that completed the 20-epoch server run.

Final server result for the compressed-context version:

| Epoch | R1 | R5 | R10 | mAP |
| --- | ---: | ---: | ---: | ---: |
| 20 | 78.8 | 88.6 | 89.9 | 58.0 |

Large artifacts are intentionally excluded from Git:

- V-JEPA cache tensors.
- model checkpoints.
- training logs and run directories.
