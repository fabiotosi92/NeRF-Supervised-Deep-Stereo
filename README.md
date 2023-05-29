
<h1 align="center"> NeRF-Supervised Deep Stereo (CVPR 2023) </h1>


<br>

This repository contains download links to our dataset, code snippets, and trained deep stereo models of our work  "**NeRF-Supervised Deep Stereo**",  [CVPR 2023](https://cvpr2023.thecvf.com/)
 
by [Fabio Tosi](https://fabiotosi92.github.io/)<sup>1</sup>, [Alessio Tonioni](https://alessiotonioni.github.io/)<sup>2</sup>, [Daniele De Gregorio](https://www.eyecan.ai/)<sup>3</sup> and [Matteo Poggi](https://mattpoggi.github.io/)<sup>1</sup>

University of Bologna<sup>1</sup>,  Google Inc.<sup>2</sup>,  Eyecan.ai<sup>3</sup>


<h2 align="center"> 

[Project Page](https://nerfstereo.github.io/) | [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Tosi_NeRF-Supervised_Deep_Stereo_CVPR_2023_paper.pdf) |  [Supplementary](https://github.com/fabiotosi92/NeRF-Supervised-Deep-Stereo/raw/main/assets/Tosi_et_al_CVPR2023_supplementary.pdf) | [Poster](https://github.com/fabiotosi92/NeRF-Supervised-Deep-Stereo/raw/main/assets/Tosi_et_al_CVPR2023_poster.pdf) | [Video](https://www.youtube.com/watch?v=m7dqHkxb4yg&t=9s) 

</h2>

<h4 align="center">


<ins>Dataset, code snippets, and trained models coming soon! </ins>

</h4>

![Alt text](./images/framework.png "architecture")


**Contributions:** 

* A novel paradigm for **collecting** and **generating stereo training data** using **neural rendering** and <ins>a collection of user-collected image sequences</ins>. Our methodology offers a means to train any stereo network using readily available user-collected images, thereby **eliminating the requirement for synthetic datasets, ground-truth depth, or (even) real stereo pairs**!

* A **NeRF-Supervised (NS) training protocol** that combines rendered image triplets and depth maps to address occlusions and enhance fine details. 

* **State-of-the art, zero-shot generalization** results on challenging stereo datasets, without exploiting any ground-truth or real stereo pair.


If you find this code useful in your research, please cite:

```shell
@InProceedings{Tosi_2023_CVPR,
    author    = {Tosi, Fabio and Tonioni, Alessio and De Gregorio, Daniele and Poggi, Matteo},
    title     = {NeRF-Supervised Deep Stereo},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {855-866}
}
```



## Dataset

We collect a total of 270 high-resolution (8Mpx) scenes in both indoor and outdoor environments using **standard camera-equipped smartphones**. For each scene, we focus on a/some specific object(s) and acquire 100 images from different viewpoints, ensuring that the scenery is completely static. The acquisition protocol involves a set of either front-facing or 360° views.

<p float="left">
  <img src="./images/dataset.png" width="800" />
</p>

**Examples of scenes in our dataset.** Here we report individual examples derived from 30 different scenes that comprise our dataset.

<ins>Download link COMING SOON!<ins>

## Pretrained Models

Here, you can download the weights of [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo)  and [PSMNet](https://github.com/JiaRenChang/PSMNet) achitectures, trained (from scratch) on rendered triplets of our real-world datasert using our NeRF-Supervised training loss. 

<ins>Download link COMING SOON!</ins>

## Qualitative Results

<p float="left">
  <img src="./images/rendered_triplet.png" width="800" />
</p>


**Examples of Rendered Images and Depth from NeRF.** We show examples on a scene of our dataset. In each case, the leftmost and rightmost columns show the rendered left and right images in a triplet, respectively. These images were obtained using small, medium, and large baselines, as indicated by the red, green, and blue lines. The center column, from top to bottom, shows the center image in the triplet, its corresponding rendered disparity map, and ambient occlusion map. Here, we adopt the Instant-NGP framework to render images.

<br>

<p float="left">
  <img src="./images/loss_comparison.png" width="800" />
</p>

**Effect of Training Losses.** From left to right: reference image, disparity maps computed by the RAFT-Stereo network trained using the popular binocular photometric loss between two images of a rectified stereo pair, the triplet photometric loss between three horizontally aligned images, the proxy-supervised loss from Aleotti et al., ECCV 2020 and, finally, our proposed NeRF-Supervised loss.  Please zoom-in to better perceive fine-details. 


<br>

<p float="left">
  <img src="./images/comparison_with_MfS.png" width="800" />
</p>

**Qualitative Comparison on Midd-A H (top) and Midd-21 (bottom).** From left to right: left images and disparity maps by RAFT-Stereo models, respectively trained with MfS or NS. Under each disparity map, the percentage of pixels with error > 2.


## Contacts

For questions, please send an email to fabio.tosi5@unibo.it or m.poggi@unibo.it

(*) *This is not an officially supported Google product.* 