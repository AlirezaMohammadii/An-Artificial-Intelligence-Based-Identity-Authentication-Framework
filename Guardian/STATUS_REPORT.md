# Guardian Repository Status — 2025-12-03

## Remote snapshot
- `origin/main` @ `25b7f31` ("Update Pitch_Signal_Recognition.py")
- `origin/AlirezaMohammadii-patch-1` @ `91fdeaa` (continuation of preprocessing updates)
- Local branch `main` has extensive uncommitted work and diverges from both remotes.

## High-impact differences vs `origin/main`
1. **Missing legacy orchestration scripts** – root-level drivers such as `Guardian/ConvModelGuardian.py`, `Guardian/CountFiles.py`, `Guardian/GvoiceTrimmer.py`, `Guardian/Poisoned_pool.py`, `Guardian/VecToMat.py`, `Guardian/main.py`, `Guardian/move_to_victims.py`, `Guardian/new_user_register.py`, `Guardian/voiceTrimmer.py`, and their companion `read.md` documents are deleted locally. Anyone relying on those entry points must migrate to the new preprocessing flow.
2. **Data documentation removed** – the `Guardian/data/**/readme.md` files that described discriminator checkpoints, guardian splits, and dataset structure are gone, so regenerating that guidance will be necessary before publishing.
3. **Preprocessing overhaul** – `Guardian/preprocessing/Attack_Victim.py`, `main.py`, and `voiceTrimmer.py` now import `numpy` and `pydub`, drop the `new_user_generator` helper, rename victim mapping outputs to `mapping.json`, hard-code 5-sample swaps, and add new helpers for renaming and trimming victim folders. This diverges from the `origin/AlirezaMohammadii-patch-1` branch that still exposes CLI options for swap counts and modes.
4. **Guardian utilities changed** – `Guardian/guardian/utils_my_version.py` now contains checkpoint cleanup, plotting, Kaldi copy helpers, and renaming logic that used to live elsewhere. `Guardian/guardian/constants.py` tweaks checkpoint roots, so any scripts that import it will point to the new dataset layout.
5. **Requirements churn** – `Guardian/requirements.txt` shrank from 6 KB to 4.5 KB; several packages were added/removed (diff pending review) which will impact reproducibility.
6. **Model pipeline updates** – `Guardian/src/pre_process_embedding.py` now imports `guardian.utils` instead of `utils_my_version`, switches back to `authentication_model.deep_speaker_models`, drops pandas display tuning, and rewrites batching/printing logic. `pre_process_npy.py`, `save_model_modified_latest.py`, `KNN_fit_my_version.py`, and `test.py` all have edits (parameter sweeps, bug fixes not yet summarized).
7. **New README-driven workflow** – `Guardian/src/README.md` (602 lines) narrates the new 3-second sample policy, train/test splits, and a multi-step execution order (main scripts → pitch recognition → visualization). This file did not exist on `origin/main`.
8. **Visualization + analysis scripts removed** – previously committed files such as `Guardian/src/Pitch_Signal_Recognition.py`, `Guardian/src/attack_success_on_aggregated_results.py`, `Guardian/src/train_modified_latest_config.py`, and `Guardian/src/visualization_module.py` are deleted or replaced by untracked versions, so notebook-driven analyses currently have no tracked source.

## Notable untracked additions
- New preprocessing helpers: `Guardian/preprocessing/Attack_Victim_modified.py`, `config/config.yaml`, `move_directory.py`, `param.py`, `trigger_embed_complete.py`.
- Visualization + analytics assets: `Guardian/src/Confusion_matrix_F1_Calculator.ipynb`, `Visualization_edit.ipynb`, `model_performance.png`, `dummy_subdir_combined_radar_plot.png`, and thousands of PNGs in `Guardian/src/visualizations/`.
- Training/testing utilities: `Guardian/src/anomaly_detection_test.py`, `audio_augmentations.py`, `embedding_21_01_2025.py`, `embedding_24_01_2025.py`, `model_3class_latest.py`, `move_deferred_Rename_Triggered.py`, `pre_anomaly_detection.py`, `pre_anomaly_detection_config.py`, `pre_process_embedding_modified.py`, `radar_chart.py`, `save_model_17_11_24.py`, `save_model_latest_config.py`, `test_3class_24_01_25.py`, `train_3class_09_02_25_ASR_user_test.py`, `train_3class_latest.py`, `train_3class_latest_vox.py`, `train_modified_version_latest_config.py`, `triggered_calss_creator.py`, `visualization_module.py` (new variant).
- Configuration + metadata: `Guardian/src/config.ini`, `config_17_11_24.ini`, `sanitized_metadata.json`, `threshold_hubert.json`.
- Aggregated metrics: `Guardian/src/aggregated_results*.json`.

## Hygiene updates
- Added repository-level `.gitignore` to drop datasets (`Guardian/data/**`), generated caches (`Guardian/src/cache/`, `Guardian/src/output/`, visualization PNGs), trained checkpoints (`*.h5`, `*.pth`, `hubert_base*.pt`), and local environments (`venv_3.10/`, `src/venv_Guardian/`). This keeps future pushes from uploading bulky assets accidentally.

## Follow-up recommendations
1. Decide whether the legacy driver scripts removed locally should be restored or officially deprecated before publishing.
2. Align preprocessing parameters (swap counts, directory naming, JSON outputs) with the branch that will remain canonical (`origin/main` or `origin/AlirezaMohammadii-patch-1`).
3. Review the new analytics and config files, add documentation, and stage them intentionally before attempting a publish.
4. Recreate missing dataset READMEs or replace them with consolidated documentation inside `Guardian/src/README.md`.
