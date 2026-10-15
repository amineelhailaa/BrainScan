# BrainScan — Brain MRI classification with CNNs

**Individual project · October 12–16, 2026 · Owner: AMINE ELHAILAA**

Submission deadline: **Friday, October 16, 2026, before midnight (Africa/Casablanca)**.

Build a TensorFlow/Keras CNN that classifies an MRI image into `glioma`, `meningioma`, `notumor`, or `pituitary`, then expose prediction through Streamlit. This implements the assignment “BrainScan – Détection du Cancer Cérébral par CNN”. The available labels define a four-class image-classification task; they do not establish cancer malignancy or clinical diagnostic validity.

**Current status:** Day 1 notebook, environment check, audit pipeline and aligned loader are implemented. The byte audit was reproduced; dependency/network and Docker access restrictions blocked decode, plotting, near-duplicate review and loader runtime verification. Task 01 is complete; Tasks 02–05 await those checks. See [Day 1 evidence and limitations](docs/day1_data_notes.md). CNN training and prediction are not implemented.

### First code: inspect the dataset

Install the inspection dependencies, then run the script from the repository root:

```bash
python -m pip install -r requirements.txt
python -m src.audit_day1
```

The Day 1 pipeline additionally writes `reports/data_inventory.csv`, `reports/working_manifest.csv`, `reports/duplicate_audit.csv` and `reports/near_duplicate_candidates.csv`. The base inspector writes `reports/image_inventory.csv`, `reports/rejected_files.csv`, `reports/duplicate_audit.json`, `reports/dataset_summary.json`, `reports/class_counts.png` and `reports/sample_images.png`. To choose a different number of example images per class, run `python src/inspect_dataset.py --samples 8`. The original images are never changed. The script only inspects and hashes files; it does not split, resize, normalize or remove data.

### Docker development environments

Docker Compose provides two CPU-only services with separate dependency lists. Jupyter is for exploration and model training; Streamlit is for the demo. The MRI dataset is mounted read-only in both services, while notebooks, source code and reports are stored in the project folder. The web ports are bound to this computer only.

```bash
docker compose build
docker compose up -d jupyter
docker compose logs jupyter
```

Open the tokenized Jupyter URL printed in the logs. To start the demo after Docker has built both images, run `docker compose up -d streamlit` and open `http://localhost:8501`. The starter page currently previews an uploaded image; CNN inference will be connected after the model and prediction function are implemented. Stop services with `docker compose down`.

The Dockerfiles and container-specific dependency lists live in `docker/`. They share pinned CPU TensorFlow, NumPy and headless OpenCV versions, with notebook or UI libraries added to each. Docker is optional for Kaggle workflows; use the notebook directly there and build Streamlit separately when ready to demo.

## Dataset already present

Read-only inventory on October 12, 2026:

| Folder | Files |
| --- | ---: |
| `Data/glioma/` | 1,621 |
| `Data/meningioma/` | 1,645 |
| `Data/notumor/` | 2,000 |
| `Data/pituitary/` | 1,757 |
| **Total** | **7,023** |

All file extensions found were `.jpg`; total size is approximately **151.3 MiB**. This checks filenames and byte hashes, not full image decode quality.

- SHA-256 audit found **6,726 unique byte hashes**, **194 duplicate groups**, and **297 extra copies**. No exact duplicate group spans multiple class folders.
- Filenames include **5,712 `Tr-`** and **1,311 `Te-`** prefixes. **79 exact duplicate groups span both prefixes**. Their original split semantics still need verification.
- Dataset source, license, patient IDs and patient independence are **not yet verified**. Do not assume a filename is a patient identifier.
- Preserve `Data/` as raw input. Exclude invalid/duplicate records through a documented working manifest. The brief's cleaning requirement applies to the usable dataset; retaining originals makes the decisions reversible.
- Freeze train/validation/test membership before augmentation or fitting. Keep duplicate and known patient groups together. If an original split is confirmed, resolve its overlaps before training.
- Decode checks and near-duplicate review remain Day 1 tasks. Byte-identical detection does not establish that all similar images or patient overlaps have been found.

## What to learn, in order

Treat each day's learning as preparation for an immediate practical task. Spend about one hour on focused study, then apply it.

| Priority | Topic | You understand it when you can… | Day |
| --- | --- | --- | --- |
| Essential | Python image handling, NumPy, OpenCV | Explain shapes/dtypes, load images safely, distinguish BGR/RGB, resize and keep labels aligned | 1 |
| Essential | Dataset quality and leakage | Explain why duplicates/patient overlap inflate evaluation and why validation differs from test | 1–2 |
| Essential | Neural-network basics | Explain weights, forward pass, loss, gradient descent/backpropagation conceptually, Adam, batches and epochs | 2 |
| Essential | CNN architecture | Trace an image through convolution, ReLU, pooling, dropout and a four-class softmax layer | 2 |
| Essential | Preprocessing and labels | Normalize exactly once and match one-hot/integer labels to the right loss function | 2 |
| Essential | Training and generalization | Read learning curves, identify overfitting, use checkpoints and early stopping, compare controlled experiments | 3 |
| Essential | Classification evaluation | Interpret precision, recall, F1, macro averages, support and confusion matrices for every class | 4 |
| Essential | Saving and inference | Reload the same class mapping/preprocessing and reproduce a batch prediction with one image | 4 |
| Essential | Streamlit and reproducibility | Build upload/preview/prediction, cache the model, handle errors and explain how to run the project | 5 |
| Optional after core completion | Transfer learning, Grad-CAM, richer tuning | Compare a pretrained network or explain a visualization without replacing the required custom CNN | After core |

**Keep this week focused:** a custom CNN, honest evaluation, working Streamlit demo and clear documentation come first. Transfer learning, Grad-CAM, cloud hosting, FastAPI, MLflow and Airflow are outside the required five-day scope.

### Focused learning resources

Read the relevant sections and apply them to BrainScan; completing every tutorial is unnecessary.

- Day 1: [OpenCV image reading and display](https://docs.opencv.org/4.13.0/db/deb/tutorial_display_image.html).
- Day 2: [TensorFlow CNN tutorial](https://www.tensorflow.org/tutorials/images/cnn) and [image classification tutorial](https://www.tensorflow.org/tutorials/images/classification).
- Day 2: [scikit-learn stratified group splitting](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html) for understanding non-overlapping groups and class proportions; this is a reference, not a requirement to run full cross-validation this week.
- Day 3: [Keras training and evaluation](https://keras.io/guides/training_with_built_in_methods/), [ModelCheckpoint](https://keras.io/api/callbacks/model_checkpoint/) and [EarlyStopping](https://keras.io/api/callbacks/early_stopping/).
- Day 4: [scikit-learn classification report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html).
- Day 5: [Streamlit file uploader](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader) and [model/resource caching](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource).

## Five-day schedule

The plan assumes **29 focused hours**: six hours each Monday–Thursday and five hours Friday, leaving about one hour on Friday for fixes. Estimates include learning and hands-on work; unattended training time is additional and depends on hardware. Start full training by Wednesday and reduce experiment count if hardware is slow.

| Day | Date | Main outcome | Focused effort |
| --- | --- | --- | ---: |
| 1 — Monday | Oct 12 | Audited dataset, exploration and safe loader | 6 h |
| 2 — Tuesday | Oct 13 | Frozen splits, preprocessing and working baseline CNN | 6 h |
| 3 — Wednesday | Oct 14 | Compared experiments and frozen best checkpoint | 6 h |
| 4 — Thursday | Oct 15 | Held-out metrics, error analysis and single-image prediction | 6 h |
| 5 — Friday | Oct 16 | Working Streamlit demo, documentation and submission | 5 h + buffer |

Follow tasks in order within each day. Critical dependencies: audit → split → preprocessing → baseline → training → model selection → test evaluation → final demo. The UI skeleton can be developed once the prediction contract is defined. Keep test observations out of model selection.

## Jira organization

Space: **[BrainScan (BRAIN)](https://amineelhailaadevai.atlassian.net/issues?jql=project%20%3D%20BRAIN%20ORDER%20BY%20key%20ASC)**.

The plan follows the previous DiaPredict pattern: **5 Epics + 5 Stories + 25 numbered Tasks**. Stories and Tasks are both children of their corresponding Epic; Tasks reference their implementing Story and share a story label. This respects Jira's standard hierarchy, where Story and Task are peers.

Each issue is assigned to AMINE ELHAILAA, carries a day label and start/due date, and contains acceptance criteria. Estimated effort is recorded in the description. All planned work begins in To Do; update it with real evidence as work is completed. This README is a planning snapshot, not an automatic Jira status sync.

| Day | Epic | User story | Tasks |
| --- | --- | --- | --- |
| 1 | [BRAIN-1](https://amineelhailaadevai.atlassian.net/browse/BRAIN-1) — Data understanding and reproducible preparation | [BRAIN-6](https://amineelhailaadevai.atlassian.net/browse/BRAIN-6) — Use a trustworthy, documented MRI dataset | 01–05 |
| 2 | [BRAIN-2](https://amineelhailaadevai.atlassian.net/browse/BRAIN-2) — Leakage-aware splits and CNN baseline | [BRAIN-7](https://amineelhailaadevai.atlassian.net/browse/BRAIN-7) — Train a first CNN on consistently prepared inputs | 06–10 |
| 3 | [BRAIN-3](https://amineelhailaadevai.atlassian.net/browse/BRAIN-3) — Training, regularization and experiment comparison | [BRAIN-8](https://amineelhailaadevai.atlassian.net/browse/BRAIN-8) — Select a reproducible model using validation evidence | 11–15 |
| 4 | [BRAIN-4](https://amineelhailaadevai.atlassian.net/browse/BRAIN-4) — Held-out evaluation and image prediction | [BRAIN-9](https://amineelhailaadevai.atlassian.net/browse/BRAIN-9) — Inspect model errors and predict from one image | 16–20 |
| 5 | [BRAIN-5](https://amineelhailaadevai.atlassian.net/browse/BRAIN-5) — Streamlit, documentation and delivery | [BRAIN-10](https://amineelhailaadevai.atlassian.net/browse/BRAIN-10) — Run and explain the complete BrainScan demo | 21–25 |

## Day 1: Monday — Understand and audit the data

**Date:** 2026-10-12 · **Story:** [BRAIN-6](https://amineelhailaadevai.atlassian.net/browse/BRAIN-6) · **Planned effort:** 6 hours.

**Learn:** Image tensors and channels; NumPy shapes and dtypes; OpenCV decoding and BGR/RGB; class labels; duplicates and data leakage.

### Task 01 — Learn the image-classification workflow

- [x] [BRAIN-11](https://amineelhailaadevai.atlassian.net/browse/BRAIN-11) · **60 minutes** · Due 2026-10-12.

Review image → preprocessing → CNN → four probabilities. Explain supervised multiclass classification, tensor shape (N,H,W,C), pixels, labels, train/validation/test, and the difference between a class prediction and a diagnosis.

**Done when:** Write a short explanation in the notebook and identify all four dataset classes.

**Evidence to save:** Notebook introduction.

### Task 02 — Prepare a reproducible Python workspace

- [ ] [BRAIN-12](https://amineelhailaadevai.atlassian.net/browse/BRAIN-12) · **60 minutes** · Due 2026-10-12.

Create the project environment and requirements for TensorFlow/Keras, OpenCV, NumPy, pandas, matplotlib, seaborn, scikit-learn, Streamlit and Jupyter. Check imports, CPU/GPU availability and memory. Set seeds; organize notebook sections. Ignore raw data, environments and large generated artifacts in Git.

**Done when:** A clean environment imports the libraries; versions, hardware and seed are recorded.

**Evidence to save:** requirements.txt; .gitignore; notebook outline.

### Task 03 — Audit files and handle invalid images

- [ ] [BRAIN-13](https://amineelhailaadevai.atlassian.net/browse/BRAIN-13) · **90 minutes** · Due 2026-10-12.

Inventory paths, labels, extensions, sizes, dimensions and channels. Accept .jpeg/.jpg/.bmp/.png case-insensitively. Decode with try/except AND check for a None result from OpenCV. Log invalid files and exclude them from the working manifest; preserve raw Data. Record the original dataset URL/license if available and investigate Tr/Te filename origins.

**Done when:** Every file is accepted or rejected with a reason; source/split uncertainties are explicit.

**Evidence to save:** reports/data_inventory.csv; reports/rejected_files.csv.

### Task 04 — Explore classes, examples and duplicates

- [ ] [BRAIN-14](https://amineelhailaadevai.atlassian.net/browse/BRAIN-14) · **90 minutes** · Due 2026-10-12.

Plot class counts and representative images. Reproduce SHA-256 duplicate groups, inspect likely near-duplicates, and review patient/source metadata if available. Initial audit: 7,023 files, 6,726 unique byte hashes, 194 duplicate groups, 297 extra copies, 79 groups spanning Tr/Te prefixes, no cross-label exact duplicate groups. Document decisions without deleting originals.

**Done when:** Counts and figures are saved; duplicate handling and patient-ID availability are documented.

**Evidence to save:** reports/class_counts.png; reports/sample_images.png; reports/duplicate_audit.csv.

### Task 05 — Implement aligned image and label loading

- [ ] [BRAIN-15](https://amineelhailaadevai.atlassian.net/browse/BRAIN-15) · **60 minutes** · Due 2026-10-12.

Load accepted images with OpenCV; choose RGB 224×224 inputs and a documented resize policy. Append an image and its label together only after successful decoding. Shuffle paired indices with a fixed seed, convert to NumPy, and assert matching lengths, shapes and finite values. Keep uint8 storage where useful to avoid large float32 copies.

**Done when:** A sample batch has the expected shape and labels still match images after shuffling.

**Evidence to save:** src/preprocessing.py; notebook loading section.

## Day 2: Tuesday — Build a valid split and your first CNN

**Date:** 2026-10-13 · **Story:** [BRAIN-7](https://amineelhailaadevai.atlassian.net/browse/BRAIN-7) · **Planned effort:** 6 hours.

**Learn:** Stratification and grouping; normalization; one-hot encoding; convolution, filters, pooling, ReLU, softmax; loss, Adam, batches and epochs.

### Task 06 — Freeze train, validation and test manifests

- [ ] [BRAIN-16](https://amineelhailaadevai.atlassian.net/browse/BRAIN-16) · **90 minutes** · Due 2026-10-13.

Resolve task 04 findings before splitting. Confirm whether Tr/Te encode an official split; if confirmed, preserve its intent but resolve overlap by excluding training copies of held-out groups and deduplicating within splits. Otherwise document a seeded stratified 70/15/15 split after grouping duplicates and patients where IDs exist. Never invent patient IDs from filenames. Keep each duplicate/patient group in one split; record proportions and class counts. Do not tune using the test set.

**Done when:** No path/hash/group overlap; all four classes represented where feasible; patient independence limitations stated.

**Evidence to save:** reports/splits.csv; reports/split_audit.md.

### Task 07 — Encode labels and normalize consistently

- [ ] [BRAIN-17](https://amineelhailaadevai.atlassian.net/browse/BRAIN-17) · **60 minutes** · Due 2026-10-13.

Persist an explicit class-to-index mapping. Use one-hot labels with categorical_crossentropy, or integer labels with sparse_categorical_crossentropy consistently. Normalize pixels to float32 [0,1] exactly once. Share preprocessing across train/validation/test and later inference. Check a sample batch and memory use.

**Done when:** Shapes, range, encoding and loss agree; mapping is saved and reloadable.

**Evidence to save:** models/class_names.json; preprocessing checks.

### Task 08 — Learn and explain CNN layers

- [ ] [BRAIN-18](https://amineelhailaadevai.atlassian.net/browse/BRAIN-18) · **60 minutes** · Due 2026-10-13.

Study Conv2D, kernels, feature maps, strides/padding, MaxPooling2D, Dropout, Flatten or GlobalAveragePooling2D and Dense. Explain parameter counts, ReLU in hidden layers and four-unit softmax output. Follow the linked TensorFlow CNN tutorial, then sketch your own architecture.

**Done when:** You can explain each layer and track shapes from a 224×224×3 image to four scores.

**Evidence to save:** Notebook architecture notes.

### Task 09 — Implement and compile the baseline CNN

- [ ] [BRAIN-19](https://amineelhailaadevai.atlassian.net/browse/BRAIN-19) · **90 minutes** · Due 2026-10-13.

Build a modest CNN with Conv2D + MaxPooling + Dropout + Dense, ending in Dense(4, softmax). Compile with Adam and the matching cross-entropy loss. Start with learning rate 1e-3, batch 32 (16 if memory constrained), max 20 epochs; these are baseline choices, not claimed optima. Save model.summary() and plot_model(); resolve pydot/Graphviz dependencies if required.

**Done when:** Architecture summary/diagram exist and a forward pass returns four finite probabilities per image.

**Evidence to save:** src/model.py; reports/model_summary.txt; reports/model_architecture.png.

### Task 10 — Run a training smoke check

- [ ] [BRAIN-20](https://amineelhailaadevai.atlassian.net/browse/BRAIN-20) · **60 minutes** · Due 2026-10-13.

Fit for 1–2 epochs on a small training subset with validation. Check label alignment, loss, output shape and memory; try overfitting a tiny training batch as a debugging check if necessary. Measure elapsed training time using time. Keep this separate from scored experiments.

**Done when:** Training completes, loss is finite and the run configuration/duration are recorded.

**Evidence to save:** Notebook smoke-run results; training entry point.

## Day 3: Wednesday — Train, diagnose and improve

**Date:** 2026-10-14 · **Story:** [BRAIN-8](https://amineelhailaadevai.atlassian.net/browse/BRAIN-8) · **Planned effort:** 6 hours.

**Learn:** Overfitting/underfitting; learning curves; augmentation; class weights; checkpoints; early stopping; learning rate and controlled comparisons.

### Task 11 — Learn overfitting and callbacks

- [ ] [BRAIN-21](https://amineelhailaadevai.atlassian.net/browse/BRAIN-21) · **60 minutes** · Due 2026-10-14.

Read train/validation accuracy and loss curves. Explain why training accuracy alone is insufficient. Study ModelCheckpoint(save_best_only=True, monitor='val_loss') and EarlyStopping(restore_best_weights=True). Explain what an epoch, batch size and learning rate change.

**Done when:** Write how to recognize overfitting and how the best checkpoint is selected.

**Evidence to save:** Notebook learning notes.

### Task 12 — Train and save the complete baseline

- [ ] [BRAIN-22](https://amineelhailaadevai.atlassian.net/browse/BRAIN-22) · **90 minutes** · Due 2026-10-14.

Train on the fixed training split with validation only. Use ModelCheckpoint and EarlyStopping; record seed, architecture, preprocessing, hyperparameters, hardware and duration with time. Save history, best model and configuration. Record a simple majority-class baseline for context.

**Done when:** A reloadable best checkpoint, learning history and complete run configuration exist.

**Evidence to save:** models/baseline.keras; reports/history_baseline.csv; reports/experiments.csv.

### Task 13 — Address generalization and class balance

- [ ] [BRAIN-23](https://amineelhailaadevai.atlassian.net/browse/BRAIN-23) · **60 minutes** · Due 2026-10-14.

Inspect training class proportions after cleaning; the raw largest/smallest ratio is only about 1.23, so justify any balancing. If needed compute class weights from training only. Try modest plausible image augmentation on training only; visually check it. Never augment validation/test or duplicate samples before splitting.

**Done when:** Chosen regularization/balancing policy has a written reason and augmentation examples look valid.

**Evidence to save:** Training configuration; augmentation preview.

### Task 14 — Compare a small set of CNN experiments

- [ ] [BRAIN-24](https://amineelhailaadevai.atlassian.net/browse/BRAIN-24) · **90 minutes** · Due 2026-10-14.

Run the baseline plus at most two controlled variants, such as dropout/augmentation or a smaller learning rate. Reuse splits and evaluation code. Compare validation macro-F1, per-class recall, loss, time and model size; record failures as well as successes. Reduce run budget if CPU training is slow; keep test untouched.

**Done when:** A comparison table identifies changes and a justified winning run.

**Evidence to save:** reports/experiments.csv; validation comparison.

### Task 15 — Freeze the selected model and artifacts

- [ ] [BRAIN-25](https://amineelhailaadevai.atlassian.net/browse/BRAIN-25) · **60 minutes** · Due 2026-10-14.

Choose the model using a predeclared validation criterion (macro-F1, then validation loss as a tie-break). Reload its best checkpoint, plot accuracy/loss and save class mapping plus preprocessing/configuration. Save a .keras model and a .h5 export to meet the brief; verify both reload consistently.

**Done when:** The final candidate is frozen with traceable configuration and reload checks.

**Evidence to save:** models/best_model.keras; models/best_model.h5; models/preprocessing.json; reports/learning_curves.png.

## Day 4: Thursday — Evaluate and build prediction

**Date:** 2026-10-15 · **Story:** [BRAIN-9](https://amineelhailaadevai.atlassian.net/browse/BRAIN-9) · **Planned effort:** 6 hours.

**Learn:** Precision, recall, F1, support and macro averages; confusion matrices; false negatives; serialization; batch dimensions; training/inference consistency.

### Task 16 — Learn multiclass evaluation

- [ ] [BRAIN-26](https://amineelhailaadevai.atlassian.net/browse/BRAIN-26) · **60 minutes** · Due 2026-10-15.

Explain accuracy, per-class precision/recall/F1, support, macro versus weighted averages and confusion-matrix axes. Distinguish predicted notumor from a medical all-clear. Plan to report tumor-class images confused with notumor without claiming clinical validation.

**Done when:** You can interpret one confusion-matrix row and explain why macro-F1 complements accuracy.

**Evidence to save:** Notebook evaluation notes.

### Task 17 — Evaluate the frozen model on held-out test data

- [ ] [BRAIN-27](https://amineelhailaadevai.atlassian.net/browse/BRAIN-27) · **90 minutes** · Due 2026-10-15.

Evaluate the selected checkpoint once on the untouched test manifest. Produce accuracy, macro-F1, per-class precision/recall/F1/support, raw and normalized confusion matrices. Keep label order fixed. Save predictions with paths and true/predicted labels. Compare against the majority baseline; record the actual scores honestly.

**Done when:** Metrics can be regenerated from saved predictions and the fixed checkpoint; no test-driven retuning.

**Evidence to save:** reports/test_metrics.json; reports/classification_report.csv; reports/confusion_matrix.png; reports/test_predictions.csv.

### Task 18 — Analyze correct and incorrect predictions

- [ ] [BRAIN-28](https://amineelhailaadevai.atlassian.net/browse/BRAIN-28) · **60 minutes** · Due 2026-10-15.

Display correct and incorrect examples across classes with model scores. Review tumor→notumor errors, image quality, duplicates/source artifacts and class confusions. Document dataset/patient-independence limitations and potential improvements for future work without changing the model using these test observations.

**Done when:** A readable error analysis includes real examples and evidence-backed limitations.

**Evidence to save:** reports/prediction_examples.png; reports/error_analysis.md.

### Task 19 — Implement predict_image(path)

- [ ] [BRAIN-29](https://amineelhailaadevai.atlassian.net/browse/BRAIN-29) · **90 minutes** · Due 2026-10-15.

Load the saved model, ordered class names and preprocessing configuration. Reuse training resize/color/normalization; add the batch dimension. Return predicted class and four model scores. Handle missing paths, unsupported formats and undecodable images with useful errors.

**Done when:** One valid image returns stable labels/scores; invalid inputs fail clearly.

**Evidence to save:** src/predict.py; predict_image(path).

### Task 20 — Verify inference and start the Streamlit flow

- [ ] [BRAIN-30](https://amineelhailaadevai.atlassian.net/browse/BRAIN-30) · **60 minutes** · Due 2026-10-15.

Check prediction agreement before/after model reload and between batch/single-image paths within numerical tolerance. Check grayscale/RGB handling and invalid files. Create a minimal Streamlit upload → preview → prediction skeleton using shared preprocessing; cache the loaded model.

**Done when:** Inference consistency checks pass and the interface can call the shared prediction code.

**Evidence to save:** tests/test_inference.py; app.py skeleton.

## Day 5: Friday — Finish the demo and submit

**Date:** 2026-10-16 · **Story:** [BRAIN-10](https://amineelhailaadevai.atlassian.net/browse/BRAIN-10) · **Planned effort:** 5 hours.

**Learn:** Streamlit uploads and model caching; reproducible execution; concise technical explanation; code-review and live-change practice.

### Task 21 — Finish the Streamlit interface

- [ ] [BRAIN-31](https://amineelhailaadevai.atlassian.net/browse/BRAIN-31) · **90 minutes** · Due 2026-10-16.

Add image upload, preview, prediction button, predicted dataset class and four scores. Validate inputs and handle missing model/errors. Use st.cache_resource for model loading. Label outputs as model scores and the app as an educational prototype, not a validated diagnosis.

**Done when:** The app works with each supported image type and fails gracefully on a corrupt/unsupported file.

**Evidence to save:** app.py; demo screenshots.

### Task 22 — Verify reproducibility end to end

- [ ] [BRAIN-32](https://amineelhailaadevai.atlassian.net/browse/BRAIN-32) · **60 minutes** · Due 2026-10-16.

From a fresh environment, follow the documented setup and run a sample prediction plus Streamlit launch. Verify the model, mapping and preprocessing files are available. Rerun focused checks for split overlap, invalid images and inference agreement. Document package versions and hardware limits.

**Done when:** A reviewer can launch the demo without relying on notebook state or your machine-specific paths.

**Evidence to save:** Reproduction checklist and verification evidence.

### Task 23 — Complete the README and project evidence

- [ ] [BRAIN-33](https://amineelhailaadevai.atlassian.net/browse/BRAIN-33) · **60 minutes** · Due 2026-10-16.

Update this README with actual installation/training/prediction/app commands, architecture, dataset source/license, split policy, metrics, experiment decisions and limitations. Add artifact paths and GitHub/Jira links. Keep data and large models out of normal Git history; provide approved acquisition/retrieval instructions.

**Done when:** README matches the delivered code and contains actual results; no claimed commands point to missing scripts.

**Evidence to save:** README.md; final figures; artifact retrieval instructions.

### Task 24 — Rehearse the 30-minute evaluation

- [ ] [BRAIN-34](https://amineelhailaadevai.atlassian.net/browse/BRAIN-34) · **60 minutes** · Due 2026-10-16.

Practice a 10-minute demo: problem/data (2m), preprocessing/model (3m), results/errors (2m), Streamlit (2m), limits/next steps (1m). Prepare for 10m code review and 10m live change. Explain normalization, split leakage, softmax/loss, dropout, callbacks and false negatives; practice changing a hyperparameter and diagnosing an invalid image.

**Done when:** One timed rehearsal completed and key choices can be explained without reading code aloud.

**Evidence to save:** Demo outline; review questions.

### Task 25 — Publish source and submit deliverables

- [ ] [BRAIN-35](https://amineelhailaadevai.atlassian.net/browse/BRAIN-35) · **30 minutes** · Due 2026-10-16.

Check the GitHub repository contains source, dependency file, documentation and planning, with no raw dataset or secrets unintentionally tracked. Confirm model retrieval and demo instructions, update Jira with evidence and actual statuses, then submit the required links before midnight on 16 October 2026, Africa/Casablanca.

**Done when:** Submission links are accessible, required outputs are present and submission is recorded.

**Evidence to save:** GitHub source; Jira planning; submission confirmation.

## Proposed project structure

The following is the original target structure. Day 1 source and notebook now exist; see the execution record above for verification status. Later-day model, training and prediction files remain planned.

```text
BrainScan/
├── Data/                         # Existing raw images, preserved locally
├── README.md                     # Learning guide, daily tasks and delivery instructions
├── requirements.txt              # To create: tested dependency versions
├── .gitignore                    # To create: data, environment and generated artifact exclusions
├── notebooks/
│   └── brainscan.ipynb            # To create: exploration, training and evaluation narrative
├── src/
│   ├── preprocessing.py          # To create: shared decode/resize/normalize logic
│   ├── model.py                  # To create: CNN construction
│   ├── train.py                  # To create: reproducible training entry point
│   └── predict.py                # To create: predict_image(path)
├── app.py                        # To create: Streamlit interface
├── models/                       # To create: checkpoints, class names and configuration
├── reports/
│   ├── initial_data_audit.json   # Present: read-only byte-hash audit
│   └── ...                       # To create: manifests, figures, experiment and test results
├── tests/
│   └── test_inference.py          # To create: meaningful inference and error-handling checks
└── docs/
    ├── project_plan.json          # Structured version of this task plan
    └── jira_issues.json           # Created Jira issue keys and their plan mapping
```

## Implementation decisions to understand

- **Input contract:** start with RGB 224×224; document how aspect ratio is handled. OpenCV color decoding needs BGR→RGB conversion. Use the same policy at inference.
- **Memory:** all 7,023 RGB 224×224 images as float32 take roughly 3.94 GiB before extra copies. Keep uint8 arrays or load batches if memory is tight; demonstrate the requested aligned NumPy image/label arrays without creating unnecessary copies.
- **Output contract:** four softmax units, saved label order, cross-entropy matching the chosen encoding. Do not use a single sigmoid output for this four-class task.
- **Baseline settings:** Adam learning rate 0.001, batch 32 (16 if needed), maximum 20 epochs, with validation-based checkpointing and early stopping. Tune a small number of choices; these are starting values.
- **Model selection:** validation macro-F1, with validation loss as a tie-break; select within each run using its best validation-loss checkpoint. Test metrics are reported after the final candidate is frozen.
- **Class balance:** inspect counts after cleaning/splitting. Use training-only class weights if evidence justifies them; automatic oversampling is not required.
- **Artifacts:** save a preferred `.keras` checkpoint and a verified `.h5` export for the brief. Save label order, preprocessing configuration and environment versions alongside them.
- **Scope:** report actual measured results. The brief does not provide a numerical accuracy threshold; do not invent one or promise clinical performance.

## Definition of done

- [ ] Dataset source and cleaning decisions are documented; invalid files and duplicates are handled reproducibly.
- [ ] Class-count chart, sample images, aligned NumPy arrays, numeric labels and normalized inputs are demonstrated.
- [ ] Train/validation/test splits are frozen, checked for overlap and described with their limitations.
- [ ] Required CNN layers, activation/loss choices, summary and architecture diagram are explained.
- [ ] Training configuration, duration, history, checkpoints and controlled experiment comparison are saved.
- [ ] Held-out accuracy, macro-F1, per-class report, confusion matrices and prediction examples are reported.
- [ ] Saved model reloads; `predict_image(path)` uses identical preprocessing and handles invalid input.
- [ ] Streamlit upload and prediction work with useful feedback on errors.
- [ ] Actual setup and run instructions have been verified from a clean environment.
- [ ] GitHub source, planning, artifact retrieval instructions and submission links are ready.
- [ ] A 10-minute demonstration, 10-minute code review and 10-minute live exercise have been rehearsed.

## Running the project

Day 1 entry points (run from the repository root with Python 3.11 and the dependencies in `requirements.txt`):

```bash
python -m src.check_environment
python -m src.audit_day1
python -m unittest discover -s tests -v
jupyter lab notebooks/brainscan.ipynb
```

The dependency-free `python -m src.audit_bytes` was verified here. The commands above require runtime verification in an environment with the dependencies installed; see [the execution record](docs/day1_data_notes.md). No training or prediction results are claimed.

## If time becomes tight

Finish one working custom CNN, a trustworthy split, required evaluation, single-image prediction and Streamlit first. Reduce extra experiment count and use the Friday buffer for fixes. Keep the required deliverables; postpone optional transfer learning, explanations and deployment infrastructure. Record incomplete work honestly.

## Questions to prepare for the evaluation

1. Why do the four folder names require four output classes?
2. What changes between a raw image and the tensor passed to the CNN?
3. Why must images and labels be shuffled together?
4. How do convolution and pooling change shape and learn useful features?
5. Why does softmax fit this output, and which cross-entropy matches your labels?
6. How did you prevent duplicates and related images from crossing splits?
7. What do your learning curves say about overfitting?
8. Why did you choose this model and these hyperparameters?
9. What does a tumor-class image predicted as `notumor` look like in the confusion matrix?
10. How do you know Streamlit and your notebook preprocess images identically?
11. Which source/patient limitations remain, and what would external validation require?
12. Can you change a learning rate, reload a checkpoint and handle a corrupt image live?

Planning source: the supplied French assignment, the local `Data/` inventory and the existing DiaPredict Jira organization. Learning-resource links were checked while preparing this plan.

