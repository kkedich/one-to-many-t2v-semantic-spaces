---
layout: project_page
permalink: /

# You can declare accent colors here
# accent: '#D21111'
# accent: darkorange

title: "On the Analysis of the One-to-Many Mapping in Cross-Modality Text-to-Video Generation with Semantic Spaces"
authors:
    - name: Karina O. M. Bogdan
      affiliation: 1
    - name: Adín Ramírez Rivera
      link: https://www.mn.uio.no/ifi/english/people/aca/adinr/
      affiliation: 2
affiliations:
    - name: Institute of Computing, University of Campinas (Unicamp)
      link: https://ic.unicamp.br/en/
    - name: Department of Informatics, University of Oslo
      link: https://www.mn.uio.no/ifi/english/
paper: https://openreview.net/forum?id=Khl7xjzTak
# video: https://www.youtube.com/@UniOslo
code: https://github.com/kkedich/one-to-many-t2v-semantic-spaces
# data: https://huggingface.co/docs/

abstract: >-
    Despite recent advances in text-to-video generation, the role of text and video latent spaces in learning a semantically shared representation remains underexplored.
    In this cross-modality generation task, most methods rely on conditioning the video generation process by injecting the text representation into it, rather than exploring the implicit shared knowledge between the modalities.
    However, the feature-based alignment of both modalities is not straightforward, especially for the <em>one-to-many</em> mapping scenario in which one text can be mapped to several valid semantically aligned videos, a challenge that generally produces a representation collapse in the alignment phase.
    In this work, we investigate and give insights into how both modalities cope in a shared semantic space where each modality representation is previously learned in an unsupervised way.
    We explore this from a latent space learning perspective with a plug-and-play framework that adopts autoencoder-based models that could be used with other representations.
    We show that the one-to-many case requires different alignment strategies than those commonly used in the literature, which struggle to align both modalities in a semantically shared space.
---

![Paper Overview]({{ site.image_base_path | append: "one_to_many_teaser.gif" | absolute_url }}){: style="display: block; margin: 0 auto;"}

## The problem
Text-to-video models learn from pairs of captions and clips. Training pipelines usually cut long videos into short chunks of 10 to 256 frames, and each chunk inherits the caption of the video it came from. So one sentence, such as "the digit 7 is moving left and right," ends up paired with many different clips, and all of them are correct. We call each such group of clips a *bucket*. In Moving MNIST (SyncDraw-MM) the largest bucket holds 1,161 clips. In the KTH human action data set it holds 406, and even the web-scale Panda70M has a bucket of 505 clips.

Most alignment methods borrow their recipe from retrieval models like CLIP, which assume that each caption has exactly one matching clip. A model trained this way has no single target to reach when it meets a bucket. It tends to settle on the average of the bucket, on its most frequent member, or on some nearby point that matches none of them. This failure is called representation collapse, and the setting that causes it is the *one-to-many* mapping that this paper studies.

![One caption, many valid videos]({{ site.image_base_path | append: "one-to-many-teaser.png" | absolute_url }})

{: .figure-caption}
**Figure 1: One caption, many valid videos.** The two left panels show where two video encoders, the pre-trained VideoMAE-v2 and our 3DConv-Base autoencoder, place every clip captioned "the digit 7 is moving left and right." Each dot is one clip, flattened to two dimensions with PCA. The right panel shows the task an alignment model faces. A single text embedding has to point to a whole cloud of correct answers instead of to one.

## Our approach
Strong text and video encoders already exist, each trained on its own modality without paired labels. We ask whether we can keep them, learn only the bridge between them, and understand what that bridge does when captions repeat. The framework is plug-and-play, so either encoder can be swapped for another.

- **Separate encoders** handle each modality. A pre-trained CLIP text encoder turns captions into embeddings. A video autoencoder (a Wasserstein autoencoder trained on clips alone) turns clips into compact codes and turns codes back into video.
- **Progressive mapping** builds the bridge in two steps. Text first goes to a shared semantic space that video codes are also mapped into. A second network then carries the shared code into the video code space, where the video decoder renders it. Training regularizes both spaces.
- **A bucket loss** trains the shared space. It counts every clip in a bucket as a positive match for its caption and clips from other buckets as negatives. The clip that was actually paired with the caption gets extra weight, which keeps the model from sending every caption in a bucket to the same point.

![Pipeline]({{ site.image_base_path | append: "pipeline.png" | absolute_url }})

{: .figure-caption}
**Figure 2: The pipeline.** A text description (left) passes through the text encoder and a mapping function into the shared semantic space (bottom right). A shared decoder carries it into the code space of the video autoencoder (top), and the video decoder renders the clip. Discriminators check the generated videos and the codes in both latent spaces.

We also adapt the evaluation. Pixel metrics such as PSNR, SSIM, LPIPS, and DISTS normally compare a generated video with one ground-truth clip, which punishes a model for producing a different but equally valid clip. Our bucket versions compare the generated video with every clip in its bucket and keep the best match. Alongside the distribution metrics FVD and KVD, we inspect the latent spaces directly. We use PCA and t-SNE plots, Chamfer and Hausdorff distances to measure how far the generated codes sit from the real ones, and a uniformity score that measures how spread out a set of codes is and so flags collapse.

To isolate the effect of alignment, we re-implement the alignment parts of CLIP, ImageBind, and CoDi on top of the same encoders and compare them with our approach. The experiments use Moving MNIST, KTH, and the TACoS cooking videos, plus Panda70M for the latent space analysis, all as 16-frame clips at 64×64 pixels.

## What we found

### Text and video spaces look very different
Captions form a few tight, separate clusters, one per meaning, while the clips behind each caption spread into dense clouds. Going from text to video therefore means fanning one point out into a whole region. How hard that is depends on the data. Moving MNIST, with few captions and big buckets, is the hardest, followed by KTH, with TACoS and Panda70M the easiest. Size alone does not make a data set easier. Panda70M is far larger than TACoS but no easier, because it has more videos per bucket on average.

![Text and video latent spaces]({{ site.image_base_path | append: "text-video-spaces.png" | absolute_url }})

{: .figure-caption}
**Figure 3: Text and video spaces.** Caption embeddings from CLIP (top, orange) and clip embeddings from VideoMAE-v2 (bottom, green) for four data sets, flattened with PCA. On Moving MNIST and KTH the captions shrink to a handful of points while the videos fill wide regions. On TACoS and Panda70M, where captions are richer and more varied, the text space spreads out too.

### The video encoder shapes alignment
We trained two video autoencoders on the same data with the same settings. UNetLDM, adapted from the latent diffusion backbone, reconstructs videos best. The simpler 3DConv-Base reconstructs worse and sometimes mixes up the digits 1 and 7. Yet 3DConv-Base gives the better alignment with text. UNetLDM spreads its codes more sparsely, and on TACoS its aligned model collapsed completely. Good reconstructions and a good alignment target turn out to be different properties.

### Two steps tends to enable better alignment than one
Mapping text through the shared space landed the generated codes closer to the real video codes than single linear projections did, as measured by lower Chamfer and Hausdorff distances. Removing the shared space and mapping text straight to video codes produced tight clusters outside the region of real videos. Without the shared step, the standard contrastive losses (InfoNCE and the multi-positive StableRep) spread the generated codes out but placed them in the wrong region. The bucket loss placed them closer but bunched them together. Self-supervised objectives adapted to the cross-modal setting (VICReg, BYOL, and SimSiam) collapsed more in the shared space.

On the video metrics, our model with 3DConv-Base scored best on every metric on TACoS, with an FVD of 27.6 against 33.8 to 37.9 for the CLIP-, ImageBind-, and CoDi-style alignments. It also had the best FVD on Moving MNIST. The winner still changes with the data set, and on KTH our UNetLDM variant had the best FVD and KVD. But video quality alone cannot be used to evaluate alignment, requiring complementary assessment.

### The usual metrics miss much of this
Video quality scores are close across methods, while the latent space plots show large differences. Part of the reason is the decoder. A caption mapped near the region of real video codes, but not inside it, asks the decoder to render a point it never saw during training. The result is a partly correct, often incomplete video, and different wrong videos can earn similar scores. None of the methods we tested, ours included, fully solves the one-to-many problem.

## What this means in practice
- Check how often captions repeat across clips before training an alignment model. Chunking long videos quietly creates buckets.
- Treat clips that share a caption as positives. Standard InfoNCE treats them as negatives.
- Choose the video encoder for the shape of its latent space, not only for its reconstructions.
- Look at the latent space. PCA and t-SNE plots, set distances, and uniformity reveal collapse that FVD and pixel metrics hide.
- Score text-to-video outputs against the whole bucket rather than a single reference clip.

## Citation
{% raw %}
```
@article{bogdan2026onetomany,
  title   = {On the Analysis of the One-to-Many Mapping in Cross-Modality Text-to-Video Generation with Semantic Spaces},
  author  = {Bogdan, Karina O. M. and Ramírez Rivera, Adín},
  journal = {Transactions on Machine Learning Research (TMLR)},
  year    = {2026},
  url     = {https://openreview.net/forum?id=Khl7xjzTak}
}
```
{% endraw %}
