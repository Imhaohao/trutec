# Coordinator review of external research

Treat the agents' research reports as candidate leads, not benchmark evidence. No researched model has been measured on this challenge yet. Ignore claims of guaranteed texture preservation, training duration, inference memory, or target attainment until verified on compute-box.

Primary sources independently checked on October 11, 2026:

- [DINOv2-small model card](https://huggingface.co/facebook/dinov2-small): Apache-2.0 metadata; feature extractor without a fine-tuned segmentation head. Patch descriptors require a measured native-tile comparison here.
- [One-shot Texture Segmentation](https://arxiv.org/abs/1807.02654): relevant task formulation and synthetic collage training. The linked [repository README](https://github.com/ivust/one-shot-texture-segmentation) describes data-generation code; it does not establish available pretrained weights or their license.
- [What Survives on Real Drawings](https://arxiv.org/html/2609.24565v2): relevant preprint, not established peer-reviewed work. Its 0.959 CNN figure is synthetic area-weighted hatch classification accuracy, not challenge document-macro IoU. Its four-window-per-class legend protocol differs from this challenge's single rectangular query. Its production-sheet metric is average precision on a different task/protocol. None of these numbers is our performance.

Decision: measure the simpler native texture and patch-matching baselines first. Keep lightweight learned query-conditioned matching and tiled pretrained descriptors as candidates. Do not reject or endorse an entire model family based on an unmeasured runtime estimate or a benchmark with a different task.

Grok found [HatchFinder](https://github.com/GreenMap-chan/hatch-finder), an exact-task pretrained candidate. Independently verified code commit `14b0e8e79d60b9d43d148ac71b41cedef5d94841` and the [3.5M model card](https://huggingface.co/GreenMap/hatch-finder-3.5m): both declare Apache-2.0. Hugging Face revision `762f842d1e1f05b14cab8f19cbc2ddb749a8385f` contains a 14,104,743-byte model with SHA-256 `47fcc870796d5e3f68dbdef1b3069911105585643350e66fdd6d2026eb9a762b`. Remote download matches the hash. The API reports weights published September 1, before this challenge's September 22 repository creation. The training-image list is unavailable; training overlap cannot be certified. This uncertainty will be disclosed if weights are used.

Decision: benchmark the pretrained weights with native overlapping tiles and an all-ones search mask. The weights license permits this research; unavailable training images are a provenance limitation, not evidence of overlap. No external dataset with unclear terms will be downloaded. No annotation-derived search mask is permitted at inference.
