# IIEG diverse corpus (2023–2025)

Source corpus of the thesis training and evaluation data. It holds **138 PDF publications with
usable text** from the IIEG (Instituto de Información Estadística y Geográfica de Jalisco), published
between 2023 and 2025: 75 on economy, 36 on government and security, 16 on society, 7 on geography
and environment, and 4 municipal booklets (Guadalajara, Puerto Vallarta, Mezquitic, Lagos de
Moreno). The topic distribution is unbalanced because several environmental publications are maps
without extractable text.

## Files

- `manifest.csv`: official publication page, PDF URL, title, month taken from the publication path,
  topic, series, SHA-256, pages and kept passages. `source_group` identifies the catalogue page; the
  four booklets share their own group.
- `md/*.md`: text extracted with `pdftotext -layout` and split into blocks of at least 200
  characters and 30 words. These are the files read by `data_path` in `configs/create_embeddings.yaml`.
- `similarity.csv`: passages removed as exact normalized matches or with textual similarity ≥ 0.94,
  with a reference to the kept document. The SHA-256 is that of the removed passage.
- `skipped.csv`: publications found but left out (no text, download size, or all their useful text
  repeated other documents).
- `pdf/*.pdf`: local copies of the originals (ignored by git).

The 138 documents total 4,888 pages. Out of 6,446 candidate passages from 141 PDFs, **4,640** were
kept; 474 exact matches (433 across documents, 41 within one) and 1,332 near-identical passages
across documents were removed. Kept passages: 3,020 economy, 783 government and security, 483
society, 211 municipal, 143 geography and environment. The similarity filter is a textual heuristic
and does not rule out paraphrased or equivalent content.

## Reproduce

From the repository root:

```bash
python3 collect_iieg.py
python3 -m unittest -q test_collect_iieg.py
```

Requires `curl`, `pdfinfo` and `pdftotext`; no Python packages beyond the standard library. Already
downloaded PDFs are reused; the official pages are queried only for missing copies.

## Splitting

Publications of one series (monthly bulletins, regional sheets) share template sentences. Splits
must keep a whole series in one partition — splitting by row or by file leaks near-copies from train
into test. `Tesis-Embeddings/src/split_dataset.py --group-col source_file --series-threshold 0.5`
does this.
