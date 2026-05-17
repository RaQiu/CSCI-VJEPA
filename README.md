## CSCI-VJEPA version layout

This repository is a working fork for CSCI-V + V-JEPA experiments on MEVID.

- Repository root: current dense-patch V-JEPA version. This version keeps all V-JEPA dense tokens and fuses them with CSCI-V RGB patch tokens in the 1421-token fusion path.
- `legacy_server_compressed_context/`: backup of the compressed-context version copied from the server after the 20-epoch run. This version compresses V-JEPA tokens into 4 context tokens before fusion.

Current compressed-context server result on MEVID overall retrieval: R1 78.8, R5 88.6, R10 89.9, mAP 58.0 at epoch 20.

<div align="center">

# Colors See Colors Ignore: Clothes Changing ReID with Color Disentanglement (ICCV-25 🥳)<br> [Publication](https://openaccess.thecvf.com/content/ICCV2025/html/Pathak_Colors_See_Colors_Ignore_Clothes_Changing_ReID_with_Color_Disentanglement_ICCV_2025_paper.html) | [Project Page](https://ucf-crcv.github.io/ReID/CSCI/) | [Paper](https://iccv.thecvf.com/virtual/2025/poster/2481) | [Arxiv](https://arxiv.org/pdf/2507.07230) | [Video](https://www.youtube.com/watch?v=tWMQR6tEcww&t=97s) | | [More ReID](https://ucf-crcv.github.io/ReID/)  
</div>

<!--
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/colors-see-colors-ignore-clothes-changing/person-re-identification-on-ccvid)](https://paperswithcode.com/sota/person-re-identification-on-ccvid?p=colors-see-colors-ignore-clothes-changing)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/colors-see-colors-ignore-clothes-changing/person-re-identification-on-ltcc)](https://paperswithcode.com/sota/person-re-identification-on-ltcc?p=colors-see-colors-ignore-clothes-changing)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/colors-see-colors-ignore-clothes-changing/person-re-identification-on-prcc)](https://paperswithcode.com/sota/person-re-identification-on-prcc?p=colors-see-colors-ignore-clothes-changing)
-->

## Highlights 🤯 


✨ Our work is among the first to use no clothing supervision and external annotation to improve the performance of transformer-based CC-ReID. <br/>
✨ We use colors as a proxy for appearance bias as a cheaper alternative to the clothing annotations. <br/> 
✨ We outperform traditional integer labels via our colors<br/>
✨ We also propose our novel S2A-self-attention, a middle ground between complete overlap of Appearance bias - Biometric features and no overlap (dual branches). <br/>
✨ One model which learns color Embeddings (Color see) and ignores it (Color ignore), all with just 1 additional class token. <br/>




## Results 😎
While paper reports average of 2 two runs, we have give best performing model weights, so accuracy will be higher than reported in paper. All the pretrained weights are on the right hand side under resources. All model weights hosted on [HuggingFace](https://huggingface.co/ppriyank/CSCI-CC-ReID/tree/main). Just do `wget [Copy download link]`


| Dataset | COLOR PROFILE | CC (R1) | CC (mAP) | General / SC (R1) | General / SC (mAP) | MODEL Wt. | 
| -------- | ------- | ------- |  ------- | ------- | ------- | ------- |
| LTCC | 44 (RGB-uv) | 50.3 | 25.9 | 85.2 | 49.6 | [ckpt](https://huggingface.co/ppriyank/CSCI-CC-ReID/resolve/main/ltcc%2B_Co-44-1245/eva02_l_cloth_best.pth) 
| PRCC | 41 (RGB-uv) | 66.5 | 62.3 | 100.0 | 99.4 | [ckpt-1](https://huggingface.co/ppriyank/CSCI-CC-ReID/resolve/main/prcc%2B_Co-41-1245/eva02_img_extra_token_best.pth) & [ckpt-2](https://huggingface.co/ppriyank/CSCI-CC-ReID/resolve/main/prcc-9-1245-16/eva02_img_extra_token_best.pth)
| CCVID | 49 (RGB-uv)  | 91.0 | 90.9 | 91.7 | 91.6 | [image-ckpt](https://huggingface.co/ppriyank/CSCI-CC-ReID/resolve/main/CCVID_IMG/eva02_l_cloth_best.pth) & [video-ckpt](https://huggingface.co/ppriyank/CSCI-CC-ReID/resolve/main/ccvid-49-1245/ez_eva02_vid_hybrid_extra_best.pth) 



| Dataset | COLOR PROFILE | Overall (R1) | Overall (R5) | Overall (R10) | Overall (mAP) | MODEL Wt. | 
| -------- | ------- | ------- |  ------- | ------- | ------- | ------- |
| MEVID | 17 (RGB-uv) | 79.7 | 87.7 | 89.2 | 56.7 | [image-ckpt](https://huggingface.co/ppriyank/CSCI-CC-ReID/resolve/main/MEVID_IMG/eva02_l_cloth_best.pth) & [video-ckpt](https://huggingface.co/ppriyank/CSCI-CC-ReID/resolve/main/mevid-17-1244/ez_eva02_vid_hybrid_extra_best.pth)

## How to run 🤓

[Setup](setup.md) has setup instructions for running code.
We pass color as input as color profile integer with `COLOR >= 50` (50 51 52 53 54 55 56) indicating RGB HISTOGRAM and `COLOR < 50` indicate RGB-uv histogram. These profiles vary in hyerparameters, and behind the scenes these numbers are translated in various implementations. 

#### CODE 
Training code is available in [All-Train](Script/all_train.sh), [Video-Train](Script/image_batch_newton.sh)
 and other ablations are available in [Ablations](Script/run.sh). The best performance from RGB-uv color profile is chosen (> averaged, as reported in paper).    
For Video model, **FIRST** train vanilla image model on video random frames as normal person ReID (**NO colors, NO clothes**) and then use those weights in EZ-CLIP to train the final video model.  We have provided the image weights above.
Evaluating the pretrained model weights is given in [Test](Script/test.sh)

 - **NOTE:** Please stick to the provided code, as there are many more things that are implemented as a possible research direction that may not work (buggy code?). The ones in scripts is rigrously tested, and verified. 

 - **IMPROVE ACCURACY** : If you have more memory than 48 GB, train with higher batch size to get even higher accuracy. Model has triplet loss which will improve as you increase the batch size. MEVID -> 82%,  PRCC-> 68%, CCVID --> 92%. 

## CITE 🥹

This work : "Colors See Colors Ignore" : 
```bibtex
@InProceedings{Pathak_2025_ICCV,
    author    = {Pathak, Priyank and Rawat, Yogesh S.},
    title     = {Colors See Colors Ignore: Clothes Changing ReID with Color Disentanglement},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {16797-16807}
}
```

Colors Code:
```bibtex
@inproceedings{afifi2019SIIE,
  title={Sensor-Independent Illumination Estimation for DNN Models},
  author={Afifi, Mahmoud and Brown, Michael S},
  booktitle={British Machine Vision Conference (BMVC)},
  pages={},
  year={2019}
}
```
Idea of predicting attributes (to disenagle) [RLQ](https://github.com/ppriyank/RLQ-CGAL-UBD/blob/main/README.md) :
```bibtex
@inproceedings{pathak2025coarse,
  title={Coarse Attribute Prediction with Task Agnostic Distillation for Real World Clothes Changing ReID},
  author={Pathak, Priyank and Rawat, Yogesh S},
  booktitle ={36th British Machine Vision Conference 2025, {BMVC} 2025, Sheffield, UK, November 24-27, 2025},
  year={2025},
  publisher={{BMVA} Press},
}
```
Video ReID EZ-CLIP:
```bibtex
@article{ahmad2025tl,
title={T2L: Efficient Zero-Shot Action Recognition with Temporal Token Learning},
author={Shahzad Ahmad and Sukalpa Chanda and Yogesh S Rawat},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=WvgoxpGpuU},
note={}
}
```


## Acknowledgement

Code built on [MADE-ReID](https://github.com/moon-wh/MADE) 
