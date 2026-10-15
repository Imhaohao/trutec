# Hatch matching: how v14 was built

Give it a drawing and a small box around one hatch pattern. It returns a mask of every region drawn with that same pattern.

| Public validation (57 queries, 13 documents) | v14 |
|---|---:|
| Document-macro IoU | **0.908** |
| Recall | 0.974 |
| Precision | 0.742 |
| Reviewed-negative FPR | 0.056 |

TruTec's stated aspiration is 0.75+ IoU with recall near 0.95 ([BENCHMARK.md](BENCHMARK.md)).

## The task

![A real construction drawing from the training split](docs/assets/drawing-01.png)

Hatches mark materials. One sheet can hold many similar patterns, with text and dimension lines printed over them.

## The pipeline

![Pipeline: drawing and query, clean query, stage 1, stage 2, mask](docs/readme/pipeline.svg)

Two gradient-boosted tree models, CPU only. No neural network, no pretrained weights.

## Clean the query first

![A query crop before and after cleaning: the arrowhead is removed, the hatch lines stay](docs/readme/query-cleaning.png)

Arrowheads and text inside the query box would otherwise be matched as if they were the pattern.

## Train stage 2 on honest scores

![Four folds: stage 1 trained on three, scored on the held-out fourth](docs/readme/crossfit.svg)

If stage 2 learned from stage-1 scores on drawings stage 1 had memorised, it would trust them too much.

## Make the model bigger, not the rules cleverer

![Cross-validated IoU: 0.755 for 200 rounds, 0.780 for 600 rounds and 63 leaves, 0.779 for 800 rounds](docs/readme/capacity.svg)

![IoU, precision and recall on public validation from v2 to v14](docs/readme/progress.svg)

Hand-written rules (v8 to v10) improved development scores and then tied on validation. More real data (v11, v12) and a bigger stage 2 chosen by cross-validation (v14) moved validation.

## Real data beats invented data

![A synthetic trap sheet: several look-alike patterns with one query box](research_v5/synthetic/previews/preview_00.png)

![Cross-validated precision falls as synthetic weight rises](docs/readme/synthetic.svg)

![Real-drawing IoU falls as the organizer's CAD drawings are down-weighted](docs/readme/cad.svg)

Our 600 synthetic trap sheets hurt (v13). The organizer's generated CAD drawings help, so they stay at full weight.

## Where it still fails

![Precision: 0.967 on 11 regular documents, 0.610 on 2 dense sheets](docs/readme/per-document.svg)

![Training drawing: neighbouring planting symbols with a similar diagonal hatch marked as false positives (red)](research_v5/diagnose7/figures/q-e4b8a4efcb903bf5.png)

Most errors are neighbouring regions drawn with a near-identical hatch for a different material.

## Validation examples

Left: drawing with the query box (orange). Middle: prediction. Right: green correct, red false positive, blue missed, grey correctly empty, uncoloured unscored.

![Success: stippled verge hatch along a road, IoU 1.000](artifacts/report-v14/q-dd48fe13cfa62e19.png)

![Success on a dense sheet: the red diagonal hatch without its blue neighbour, IoU 0.994](artifacts/report-v14/q-752be75351e18024.png)

![Failure on the same sheet: a small purple hatch spreads into the surrounding diagonal hatches, IoU 0.160](artifacts/report-v14/q-61012bd70ddb4b3b.png)

## Run it

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-inference.txt
curl -L -o models-v14.zip https://github.com/Imhaohao/trutec/releases/download/v14/models-v14.zip
unzip models-v14.zip -d models/v14
.venv/bin/python infer_v14.py --inputs artifacts/validation-inputs.json --data-root dataset --output-dir predictions
.venv/bin/python evaluate.py --manifest dataset/val.json --data-root dataset --predictions predictions --output metrics.json
```

A full rerun reproduces all 57 validation masks pixel for pixel. Rebuild these figures with `python tools/readme_figures.py`.

## Further reading

- [technical-report.txt](technical-report.txt): method, data use, validation exposure, timing
- [PROGRESS.md](PROGRESS.md): every version and experiment, including rejected ones, with numbers
- Challenge: [SPECIFICATION.md](SPECIFICATION.md), [DATASET.md](DATASET.md) (`python download_data.py`), [SUBMISSION.md](SUBMISSION.md)

Drawings and annotations: TruTec Hatch Matching Challenge dataset-v1 ([DATA_TERMS.md](DATA_TERMS.md)). Code: [MIT](LICENSE).
