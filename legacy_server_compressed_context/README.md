## Legacy server compressed-context version

This folder is a code backup of the CSCI-V + V-JEPA version that completed the
20-epoch MEVID run on the server.

Key difference from the repository root:

- This version uses `MODEL.JEPA_CONTEXT_TOKENS: 4`.
- V-JEPA dense tokens are projected and compressed into 4 context tokens before
  fusion.
- The repository root contains the newer dense-patch path, where all 392 V-JEPA
  tokens are fused together with 1024 CSCI-V RGB patch tokens.

Final server result for this version on MEVID overall retrieval:

| Epoch | R1 | R5 | R10 | mAP |
| --- | ---: | ---: | ---: | ---: |
| 20 | 78.8 | 88.6 | 89.9 | 58.0 |

Checkpoints and JEPA cache files are not stored in this Git repository.
