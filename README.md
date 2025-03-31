<p align="center">
<h1 align="center"><strong>GaussHDR: High Dynamic Range Gaussian Splatting via Learning Unified 3D and 2D Local Tone Mapping</strong></h1>
<h3 align="center">CVPR 2025</h3>

<p align="center">
    <a href="https://scholar.google.com/citations?hl=en&user=-moPItwAAAAJ">Jinfeng Liu</a><sup>1</sup>,</span>
    <a href="https://scholar.google.com/citations?hl=en&user=KKzKc_8AAAAJ">Lingtong Kong</a><sup>2</sup>,
    <a href="https://libraboli.github.io/">Li Bo</a><sup>2</sup>,
    <a href="https://www.danxurgb.net/">Dan Xu</a><sup>1</sup>
    <br>
        <sup>1</sup>HKUST,
        <sup>2</sup>vivo Mobile Communication Co., Ltd
</p>

<div align="center">
    <a href='https://arxiv.org/abs/2503.10143'><img src='https://img.shields.io/badge/ArXiv-Paper-b31b1b.svg'></a>  
    <a href='https://liujf1226.github.io/GaussHDR/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  
    <a href='https://drive.google.com/file/d/1SZYoikKiCvBdGnWmJ1qHtGuspub_LkV4/view?usp=drive_link'><img src='https://img.shields.io/badge/Preprocessed-Data-blue'></a>  
    <a href='https://drive.google.com/file/d/1uaBfv_9boxl9pl3IMED5WIGcbsZMjUS9/view?usp=drive_link'><img src='https://img.shields.io/badge/Pretrained-Models-orange'></a> 
</div>
</p>

<br>

![teaser](https://github.com/user-attachments/assets/ed670fcb-8e5a-4f2c-860c-46d520c584b7)

## Abstract
High dynamic range (HDR) novel view synthesis (NVS) aims to reconstruct HDR scenes by leveraging multi-view low dynamic range (LDR) images captured at different exposure levels. Current training paradigms with 3D tone mapping often result in unstable HDR reconstruction, while training with 2D tone mapping reduces the model's capacity to fit LDR images. Additionally, the global tone mapper used in existing methods can impede the learning of both HDR and LDR representations. To address these challenges, we present GaussHDR, which unifies 3D and 2D local tone mapping through 3D Gaussian splatting. Specifically, we design a residual local tone mapper for both 3D and 2D tone mapping that accepts an additional context feature as input. We then propose combining the dual LDR rendering results from both 3D and 2D local tone mapping at the loss level. Finally, recognizing that different scenes may exhibit varying balances between the dual results, we introduce uncertainty learning and use the uncertainties for adaptive modulation. Extensive experiments demonstrate that GaussHDR significantly outperforms state-of-the-art methods in both synthetic and real-world scenarios.

![framework](https://github.com/user-attachments/assets/1fd2c4ff-b372-4696-bf8e-e96b79b3e03c)


## Demo Videos
We provide demo videos of novel HDR and LDR renderings for four scenes. For more qualitative comparisons, please refer to our paper and project page.

<table align="center">
  <tr>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/4bb7cb6b-6d3a-412d-953f-0f182d7a35a9" controls width="100%"></video>
    </td>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/224be397-e289-4326-8781-fabad4cc09e8" controls width="100%"></video>
    </td>
  </tr>
  <tr>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/01d91f06-3e66-4083-b0e7-3f1ff0ca1b8f" controls width="100%"></video>
    </td>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/47343c8b-903a-410e-9f59-f9a717a3864a" controls width="100%"></video>
    </td>
  </tr>
</table>



## Codes
We will realease the codebase of GaussHDR and our preprocessed data soon. Stay tuned.



## Citation

```BibTeX
@inproceedings{gausshdr,
     title={GaussHDR: High Dynamic Range Gaussian Splatting via Learning Unified 3D and 2D Local Tone Mapping},
     author={Jinfeng Liu and Lingtong Kong and Bo Li and Dan Xu},
     booktitle={CVPR},
     year={2025}
     }
```
