# ** ACCEPTED AT NEURIPS 2025**

# Pancakes: Consistent Multi-Protocol Image Segmentation Across Biomedical Domains


Official implementation of: _Pancakes: Consistent Multi-Protocol Image Segmentation Across Biomedical Domains_.  
[Marianne Rakic](https://mariannerakic.github.io/), Siyu Gai, Etienne Chollet,
[John V. Guttag](https://people.csail.mit.edu/guttag/) \& [Adrian V. Dalca](https://www.mit.edu/~adalca/)


## Abstract
A single biomedical image can be meaningfully segmented in multiple ways, depending on the desired application. For instance, a brain MRI can be segmented according to tissue types, vascular territories, broad anatomical regions, fine-grained anatomy, or pathology, etc. Existing automatic segmentation models typically either (1) support only a single protocol -- the one they were trained on -- or (2) require labor-intensive manual prompting to specify the desired segmentation. We introduce \mymethod{}, a framework that, given a new image from a previously unseen domain, automatically generates multi-label segmentation maps for \textit{multiple} plausible protocols, while maintaining semantic consistency across related images. Pancakes introduces a new problem formulation that is not currently attainable by existing foundation models. In a series of experiments on seven held-out datasets, we demonstrate that our model can significantly outperform existing foundation models in producing several plausible whole-image segmentations, that are semantically coherent across images.
