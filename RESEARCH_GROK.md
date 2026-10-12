# External texture data and reproducible matching

Retrieval date: 2026-10-11. Every license below was read from the linked page, repository API, or file header on that date. Sizes measured with an HTTP `HEAD` are marked measured. Sizes copied from a page, without a `HEAD`, are marked stated. Sources that did not load are listed as unverified and are not used in the recommendation.

This note covers external data and classical texture matching. It does not report a challenge score. The 73.73% document-macro IoU in `BENCHMARK.md` is a historical measurement on 37 real queries across 12 documents, and it is a different suite from public validation.

## Task constraints that bind the data choice

Public training has 448 queries: 192 real queries in 62 document groups, plus 256 generated CAD queries. Public validation has 57 real queries in 13 document groups. Private test has 40 real queries in 14 groups. Real document groups are disjoint. Generated CAD is training-only. `DATA_TERMS.md` allows non-commercial research and forbids redistributing the drawings as a separate commercial product. Repository code is MIT; the dataset terms are separate.

Scoring uses the known domain after removing the query rectangle and any context rectangles. Inside that domain, annotated positives are positives and the rest of the known domain is negative. Pixels outside the known domain are unknown. Unknown is unscored. A blank label is an explicit subset of the known domain, and other-pattern negatives are not blank. A positive-only example has no negative domain. An empty-target example has null IoU. The primary number is the mean, over documents, of the mean query IoU inside each document.

`make_inputs.py` keeps `id`, `image`, `width`, `height`, `query_box`, and `context_boxes`. Inference must not open annotation masks, annotation rectangles, or `expected-counts.json`.

## License matrix

| Source | What it actually is | License evidence read on 2026-10-11 | Size | Fit for this task |
|---|---|---|---|---|
| [DTD r1.0.1](https://www.robots.ox.ac.uk/~vgg/data/dtd/index.html) | 5,640 web photos, 47 attribute classes, classification splits | Page says "research purposes." `README.txt` has no grant. [VGG group-2 terms](https://www.robots.ox.ac.uk/~vgg/terms/dataset-group-2-access.html) do not list DTD. Images came from Google and Flickr, so third-party copyrights remain. | Measured archive `625,239,812` bytes. Page says 625M. | Weak. Category labels such as `zigzagged` are not a same-drawing hatch mask. |
| [USC-SIPI textures, Brodatz portion](https://sipi.usc.edu/database/copyright.php) | Scans of prints, plus mosaics | SIPI says it does not own these copyrights and "the copyright is probably retained by the author." Mosaics are "unclear." [Stanford FAQ](https://graphics.stanford.edu/projects/texture/faq/brodatz.html) quotes a Dover three-illustration limit. [VisTex](https://vismod.media.mit.edu/vismod/imagery/VisionTexture/vistex.html) says Brodatz is not freely available for research use. | Measured `textures.tar.gz` `29,768,655` bytes; `textures.zip` `29,720,623` bytes. Page says 29.7MB. | Do not use. |
| [KTH-TIPS / TIPS2](https://www.csc.kth.se/cvap/databases/kth-tips/download.html) | Cropped 200×200 material photos under pose, scale, and light | "Freely available" on the download page and in the [TIPS guide](https://www.csc.kth.se/cvap/databases/kth-tips/doc/node1.html). No license instrument on the fetched pages. Full-size images require emailing Eric Hayman. | Stated: color TIPS 68MB, grey 23MB, TIPS2-a 374MB, TIPS2-b 386MB, extra images 12MB. Not measured. | Material classification, not hatch segmentation. |
| [CUReT](https://www.cs.columbia.edu/CAVE/software/curet/html/about.php) | 61 physical samples, BTF/BRDF, 200+ view and light combinations | "Publicly available for research purposes." No redistribution grant on the about or download pages. | Per-sample archives. Total byte size unverified. | Material appearance, not line drawings. |
| [Kylberg v1.0](https://kylberg.org/kylberg-texture-dataset-v-1-0/) | 28 classes, 160 patches of 576×576, optional rotations | Citation requested. No license grant on the page or on the [Uppsala mirror](https://user.it.uu.se/~gusky180/texture/). A third-party CC BY-NC label was not on these pages. | Stated: about 0.9 GB without rotation, 10.7 GB with rotation. Not measured. | Natural textures. Citation is not a license. |
| [VisTex](https://vismod.media.mit.edu/vismod/imagery/VisionTexture/distribution.html) | MIT texture photos and scenes, unmaintained since December 2002 | "As is." No grant on the pages fetched. | Measured `VisTex.tar.gz` `161,767,641` bytes. Page says 165 megs compressed, 240 uncompressed. | Natural textures. License missing. |
| [Extended Outex, Lille](https://color.univ-lille.fr/datasets/extended-outex/) | Classification suites, 24 or 68 textures, rotation and illuminant | Page warns that the original Outex site seems discontinued. No license text. | Not measured. | Classification patches, not query masks. |
| [Prague texture benchmark](https://mosaic.utia.cas.cz/) | Generated mosaics for segmentation ranking | No license statement on the introductory page. Custom mosaics require registration. | Generated on request. Size unverified. | Natural-texture mosaics, not construction hatches. |
| [FloorPlanCAD official](https://raw.githubusercontent.com/floorplancad/floorplancad.github.io/master/index.md) | 15,663 CAD floor plans as of the 2021-11-26 note; panoptic symbols | Annotations and website: CC BY-NC 4.0. "We do not own the copyright of the drawings." Shutdown note dated 2023-01-19. | Official archive size unverified. | Symbol and wall/parking classes. Query-conditioned hatch identity is absent. |
| [Voxel51/FloorPlanCAD](https://huggingface.co/datasets/Voxel51/FloorPlanCAD) | Hugging Face mirror, 5,314 files | Card license `cc-by-sa-4.0`. This is a different grant from the official CC BY-NC 4.0 text. | API `usedStorage` `791,931,830` bytes. | Do not treat this card as the official license. |
| [ArchCAD HF](https://huggingface.co/datasets/jackluoluo/ArchCAD) | Gated CAD slices, primitive labels | Card `cc-by-nc-4.0`, manual gate, non-commercial checkbox. | API `usedStorage` `2,631,832,494` bytes. | The public class table has Wall and no hatch class. See note below. |
| [CubiCasa5K](https://github.com/CubiCasa/CubiCasa5k) | 5,000 floor-plan photos, 80+ polygon classes | GitHub `LICENSE` says CC BY-NC 4.0. Zenodo record [2613548](https://zenodo.org/records/2613548) says `cc-by-nc-sa-4.0`. | Measured Zenodo zip `5,469,495,706` bytes. README says an LMDB cache is about 105 GB. Code expects CUDA. | Floor-plan rooms and icons. The two licenses disagree. |
| [SESYD](http://mathieu.delalandre.free.fr/projects/sesyd/) | Synthetic floor plans and diagrams for symbol spotting | Cite the 2010 IJDAR paper. No license instrument on the fetched page. | Page counts collections. Byte size unverified. | Symbol boxes, not hatch regions. |
| [hatch-finder code](https://github.com/GreenMap-chan/hatch-finder) | Reference hatch in, heatmap out. Alpha. | Apache-2.0 `LICENSE`. | GitHub repository. Sample set is 20 train and 10 validation images, in-repo. | Closest public task, with a required search mask this challenge does not supply. |
| [hatch-finder-3.5m](https://huggingface.co/GreenMap/hatch-finder-3.5m) | Weight file `model.pt` | Card and README: Apache-2.0. | Measured `model.pt` `14,104,743` bytes. | Training images for the stated 7,000 examples are not linked. Metrics are on that model's own split. |
| [hatch-finder-11m](https://huggingface.co/GreenMap/hatch-finder-11m) | Weight file `hatch_finder.pt` | HF card `apache-2.0`. | Measured `44,535,011` bytes. | Training-data card was not fetched. Provenance unverified. |
| [OS-TR](https://github.com/zhukaii/OS-TR) | IJCAI 2019 one-shot texture retrieval code | GitHub `license: null`. README says a running guide will come later. Last push 2019-09-07. | `train.py` is 7,872 bytes. Supplementary PDF is 3,188,750 bytes. No weight file in the root listing. | `train.py` calls CUDA. Dataset choice in that file is DTD. Not a CPU/MPS binary. |
| [One-shot Texture Segmentation code](https://github.com/ivust/one-shot-texture-segmentation) | Data generator for arXiv:1807.02654 | GitHub `license: null`. README is one sentence. Last push 2018-07-06. | No weight file in the README. | Paper builds 256×256 mosaics from DTD and a VGG texture model, with a 64×64 reference. Generator, not a hatch model. |
| [OpenCV 4.5+](https://opencv.org/license/) | `getGaborKernel`, `filter2D`, `dft`, `matchTemplate` | Apache 2.0. | Library, not a dataset. | Runs on CPU. Use this for the first matcher. |
| [scikit-image 0.26 docs](https://scikit-image.org/docs/stable/license.html) | `gabor`, `gabor_kernel`, `local_binary_pattern`, `graycomatrix`, `graycoprops`, `structure_tensor`, `hog`, `match_template` | Default `Files: *` is BSD-3-Clause. Those functions are not in the listed exceptions. | Library. | CPU. Same role as OpenCV. |
| [ezdxf](https://github.com/mozman/ezdxf) | DXF reader and hatch-pattern parser | MIT. | Library. | Parser only. Pattern geometry still needs a clean source. |
| [dashing](https://github.com/jepler/dashing) | C++ hatch renderer, archived | zlib. | Repo includes `HSHP21E0.pat`, `HSTON4E0.pat`, `HWOOD6E1.pat`. | Code license does not clear those pattern files. Do not train on them. |

## What each useful family can and cannot teach

**Photographed texture collections** (DTD, KTH-TIPS, CUReT, Kylberg, VisTex, Outex, Prague) label materials or perceptual attributes. A construction hatch is a repeated line spacing, orientation, and ink density drawn in one document. A photo of fabric does not supply that label. DTD is additionally a poor legal fit: the pages that host it do not carry the 12-month VGG research contract, and the pictures were collected from the web.

**Brodatz** is the clearest reject. The host says the author likely still holds the copyright, and the archive that is easy to download is not a license.

**CAD collections** label symbols, walls, rooms, and doors. FloorPlanCAD's official text licenses the annotations and says the drawings belong to someone else. The Voxel51 card's CC BY-SA 4.0 grant is not the same text. ArchCAD's paper appendix lists the layer name `S-WALL-HATCH` as "the filling or material pattern of a wall in structural drawings." That row is a layer-name example. The Hugging Face class table that was fetched (ids 0–29 and 100) contains Wall and does not contain a hatch class. Wall-fill primitives may be folded into Wall or Others; that mapping was not verified from the files. CubiCasa5K's GitHub and Zenodo licenses disagree (CC BY-NC 4.0 versus CC BY-NC-SA 4.0), and the task is floor-plan parsing.

**hatch-finder** is the only fetched project whose stated task is "given a drawing and a reference hatch, mark the matching pixels." The published 3.5M model card says it was trained on 7,000 examples at 896×896, with a search mask, and reports Dice 0.94328 on its own validation split. That number is not a TruTec measurement. The 7,000 images have no dataset URL and no data license. The README selects CUDA when present and otherwise CPU. MPS is not mentioned. The tested PyTorch build is a CUDA 12.8 nightly. A documented CPU smoke test trains one epoch on the 30-image sample. The challenge request has no search mask. Feeding a mask built from labels would leak the answer. An all-ones mask is a different task and is unmeasured.

**OS-TR and the 2018 one-shot segmenter are real papers and real repositories.** They are not shippable here. OS-TR has no license and its training script moves tensors to CUDA. The 2018 repository has no license and publishes a generator for DTD mosaics. Neither root listing contained a weight file.

**Line, frequency, and orientation tools that do run on this Mac** are already in permissively licensed libraries:

- OpenCV `getGaborKernel`: orientation `theta`, wavelength `lambd`, envelope `sigma`, aspect `gamma`.
- OpenCV `dft` and scikit-image `match_template`: spacing from the autocorrelation peak, and a normalized template check.
- scikit-image `structure_tensor`: dominant orientation and coherence.
- scikit-image `local_binary_pattern`, `graycomatrix` / `graycoprops`, and `hog`: local pattern, co-occurrence, and gradient orientation. These see ink texture. On a pure parallel hatch, the Fourier peak and the structure-tensor orientation are the quantities that name the pattern. LBP and HOG are supporting channels, because a small shift of a line grid changes a binary code.

A clean synthetic source is a renderer written for this project: parallel lines and crosshatches with chosen angle, spacing, width, and dropout. ezdxf can parse a pattern file under MIT, and dashing can render one under zlib. The vendor `.pat` files inside dashing are not cleared by those code licenses. Autodesk pattern libraries were not fetched and are not recommended.

## Facts versus hypotheses

Facts:

- Unknown pixels are outside the scored union of true positives, false positives, false negatives, and true negatives.
- Query and context rectangles are removed from the scored domain.
- Document-macro IoU averages queries inside a document, then averages documents. A large drawing does not outweigh a small one.
- Positive-only examples cannot measure rejection. Empty-target IoU is null.
- No fetched public dataset supplies same-drawing, query-conditioned hatch masks at native resolution under a license that is both explicit and internally consistent.
- Classical Gabor, Fourier, structure-tensor, LBP, GLCM, HOG, and normalized template code is available under Apache-2.0 or BSD-3-Clause and runs on CPU.

Hypotheses, unmeasured:

- Matching orientation, period, and ink density from the query crop will separate two hatches on one sheet better than a material-classification embedding.
- A tolerance of a few pixels on period will absorb scan blur. The tolerance has to be chosen on a training-document holdout.
- Eroding the known positive and known negative masks by about half the query period, and ignoring that boundary band, will keep mixed windows out of the loss.
- Context rectangles are extra inputs, and the spec does not say they are extra positives. The safe training default ignores them as exemplars and still excludes them from the loss, matching the scorer.
- Photographic texture pretraining will not move document-macro IoU on real drawings. That is a reason to skip those downloads, not a measured ablation.
- hatch-finder with an all-ones mask may fire on the right family of lines and may also fire on every similar family. Its 896×896 training size can erase line spacing when a full sheet is resized and then blown back up. The scorer will not resize the submitted mask.

## Sampling strategy

Split by `document_id` before touching public validation. Freeze the split by hashing `document_id` and holding out entire groups, about 12 of the 62 real training groups. Every query on a held-out drawing stays out, including a second hatch on the same sheet. Count the held-out queries after the split and publish that count. Generated CAD stays in the training pool and gets its own score. It never enters the real-document mean.

For each training query, build the domains in `SPECIFICATION.md` section 2. The loss sees only known pixels outside the query rectangle and context rectangles. Positives are the annotated positive set inside that domain. Negatives are the other known pixels. Unknown pixels are omitted from the loss. If a query has no known negative pixels, it contributes no negative term. Blank pixels are a reported slice of the negatives, not a third training class, unless a later holdout experiment shows that separating them changes reviewed-blank false positives.

Sample equal counts of known positive and known negative pixels per step so thin lines are not drowned by paper. The evaluator remains unweighted. Record the sampling weights as training-only.

Encode the query from the crop alone. Estimate dominant orientation, the perpendicular spacing, and the dark-pixel fraction inside the crop. Compare each drawing cell with that triple. A crosshatch matches a crosshatch query when both directions are present in the query. A single-direction query does not match a crosshatch that merely shares one direction. Do not search a wide scale pyramid: a different spacing is a different pattern. Fit the small spacing tolerance on the document holdout.

Write the full native mask. Boundary cells that mix hatch and paper are the usual error. Ignore an eroded band during training. At inference, a narrow uncertainty band can be left off unless the holdout says filling it raises recall without a matching rise in known-negative false positives.

## Leakage and score inflation

- Training and testing different queries of the same `document_id` leaks the drawing. The holdout unit is the document group.
- Public validation images stay out of training, including unlabeled. One locked evaluator run is the public number. `SUBMISSION.md` requires disclosing any use of validation labels for choices.
- `expected-counts.json` is an integrity file. Reading it to pick a threshold is leakage.
- Labeling unknown pixels as negative teaches the model the shape of the reviewed window. At test time that window is absent, so the model can look strong on a scorer that only sees known pixels and still miss the same pattern elsewhere. Experiment 3 measures this.
- Painting the whole page positive raises pooled recall and is penalized only where a negative domain exists. Report `positive_only_examples`, `empty_target_examples`, precision, and reviewed-blank false positives beside recall.
- The official scorer ignores unknown pixels, so expanding the mask into unknown paper does not raise IoU and does raise false positives where the known domain is negative.
- Thresholds, periods, and morphology chosen on public validation inflate the submitted number. Choose them on the training-document holdout.
- Averaging generated CAD into the real document-macro mixes a training-only kind into a real-only metric. Validation contains no generated CAD.
- Reporting the historical 73.73% beside a new validation run without the suite name mixes two denominators.
- Resizing the prediction, or dropping a failed file, violates the scorer. A model may resize internally and must still emit the original width and height.
- External weights whose training images are undisclosed can contain drawings related to this set. hatch-finder's 7,000-image corpus has no public file list, so it stays out of the first three experiments.

## Three experiments

Run all three on the same frozen document holdout. Publish holdout counts. Do one public-validation evaluation, of the holdout winner, after the choices are frozen. Do not claim the aspirational 0.75 / 0.95 numbers as the expected result.

1. **CPU descriptor matcher, no external images.** From the query crop, compute orientation (structure tensor), spacing (autocorrelation along the normal), ink density, and a normalized template response. OpenCV and scikit-image only. Pick the accept threshold and the spacing tolerance on the holdout. Record document-macro IoU, pooled recall, pooled precision, reviewed-blank false-positive rate, empty-target false-positive rate, eligible query and document counts, wall time, and peak resident memory. Tile large sheets so the process stays inside 16 GiB.

2. **Same features, trained only on clean pixels.** Add a renderer of parallel and crossed lines with random angle, spacing, width, and gaps. Train a small classifier on those mosaics plus the challenge training queries, loss masked to known pixels outside the query and context. Generated CAD may be included and must be scored separately. Compare with experiment 1 on the same real holdout documents. Keep CPU as the required path. Use MPS only if the installed PyTorch build documents it for that operator, and record which path ran.

3. **Unknown-as-negative control.** Repeat experiment 2 with pixels outside the known domain labeled negative. Compare document-macro IoU, pooled recall, and reviewed-blank false positives on the same holdout. This is the measurement of the supervision bug. A gain on the holdout that disappears when the known window no longer matches the test drawing is the failure this control is there to catch.

Leave OS-TR, the 2018 generator, Brodatz, DTD, and hatch-finder weights out of these three runs. A later run of hatch-finder is reasonable only after the 7,000 training images are identified and licensed, and only with an explicit search-mask policy that does not read labels.

## Unverified

Official UIUC texture pages, the original Outex host, FloorPlanCAD's full archive byte size, CUReT's total byte size, Kylberg and KTH file sizes beyond the stated figures, the hatch-finder-11m training card, and the on-disk mapping from ArchCAD layer `S-WALL-HATCH` to a semantic id. Those items are unused.

Nia was unavailable in this session (API key rejected). Memory search for this project returned no prior notes.
