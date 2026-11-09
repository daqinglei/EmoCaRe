# EmoCODA

Code for **EmoCODA**, a causal decoder for conversational emotion recognition with a multimodal large language model under arbitrary missing modalities.
EmoCODA (Causal Online Decoding with Adaptive Transitions; called EmoCaRe during development, so code, file names and result-file keys keep the old name) = MACD + GCT + CCT, on top of a fine-tuned Qwen2.5-Omni-3B:

- **MACD** calibrates the scores of each modality combination, divides by the training class prior, and runs a forward recursion over the utterance sequence of a speaker.
- **GCT** selects the transition matrix by the gap since the speaker last spoke.
- **CCT** adjusts the transition using the earlier conversation, with a small model fitted on training labels.

The paper citation will be added after publication.

## What is here

| Path | Content |
|---|---|
| `experiments/paper2_ctx_lib_*`, `paper2_ctx_transition_*`, `paper2_newmain_core_*`, `paper2_normalized_macd_*`, `paper2_rrtdV_*` | The method, its frozen selection and the leave-one-out comparison |
| `experiments/paper2_newmain_bases_*`, `paper2_ctx_matchhist_*`, baseline scripts | Other base models and baselines |
| `experiments/paper2_tables_newmain_*`, `paper2_supp_tables_newmain_*`, `paper2_newmain_holm_*`, `paper2_newmain_fig*` | Tables, Holm correction, data figures |
| `experiments/build_*`, `freeze_*`, `run_meld_qwen_*`, `paper2_session_fivefold_clipfix_*`, `paper2_m3ed_*` | Protocols, fine-tuning, scoring |
| `work_dir/*.json` | Aggregate results behind the paper's numbers; they hold no corpus text |

**Not included:** datasets, media, model weights and per-utterance scores. The dataset licenses restrict redistribution. Fig. 1 examples quote corpus text, so their files are also left out.

Script names keep their original dates so that results stay traceable. Some training helpers carry older names (`harm_screen`, `c6`, `planb`) because the training driver still calls them. Comments that cite `SSOT 10.18.x` refer to our internal research log, which is not distributed.

## 1. Rebuild the tables and figures (no GPU, no datasets)

Python 3.10 is required for the pinned versions.

```bash
python3.10 -m venv .venv && .venv/bin/pip install -r requirements-analysis.txt
PYTHON=.venv/bin/python scripts/rebuild_tables.sh
```

Output goes to `results_rebuilt/` and `paper/figures/newmain/`.

## 2. Rerun the decoder (needs the per-utterance scores from step 3; CPU)

```bash
export EMOZOOM_ROOT="$PWD" CTX_REPS=4 NEWMAIN=1
# frozen selection of the CCT weight, then the leave-one-out comparison, then MatchHIST
python experiments/paper2_ctx_transition_20261029.py meld test
python experiments/paper2_newmain_core_20261029.py meld
python experiments/paper2_ctx_matchhist_20261029.py meld
# other base models: emollama_v2 mmin mmin_official qwen7b zeroshot oppo
python experiments/paper2_newmain_bases_20261029.py emollama_v2 meld
```

Use `m3ed` in the same way. For IEMOCAP use `iemocap4` or `iemocap6` and set `EMOCARE_FINISHED=1`: the paper's IEMOCAP results read only earlier utterances that have already finished. Add `EMOCARE_PV=1` for bootstrap p values. Do not set `OMP_NUM_THREADS`.

Outputs are new timestamped files in `work_dir/`. The stored files are not overwritten.

## 3. From the official datasets to the scores (GPUs)

Set `EMOZOOM_DATA` to the folder holding the datasets, and set the Qwen2.5-Omni-3B path in the protocol's `training.model` field. Obtain each dataset from its owners under its license.

| Dataset | Steps, in order |
|---|---|
| MELD | `freeze_meld_qwen_sft_platform_protocol.py`, `paper2_build_meld_official_splits_20260916.py`, `paper2_meld_emp_devselect_test_20260916.py` (the paper uses its control arm; the other arm is not part of EmoCODA) |
| IEMOCAP | `build_iemocap_qwen_protocol.py`, `paper2_iemocap_clips_v2_20261010.py`, `paper2_build_session_protocol_clipfix_20261010.py`, `paper2_session_fivefold_clipfix_20261010.py` with `TARGET=iemocap4` or `iemocap6` |
| M3ED | `paper2_build_m3ed_at_protocol_20260922.py` (media under `data/M3ED`), `paper2_m3ed_train_queue_20260922.sh`, `paper2_m3ed_eval_queue_20260922.sh` |

These are the scripts as they were used, written for a four-GPU server. Expect to adapt the GPU settings.

## What was verified

| Check | Result |
|---|---|
| Rebuild from this repository in a clean Python 3.10 environment, without any data | Tables, Holm correction and figures build; the main-table row of EmoCODA equals the paper |
| Rerun with this code and the private scores: frozen selection on the four settings | The selected weights equal the stored ones |
| Rerun: leave-one-out comparison, MatchHIST and every base model, 26 result groups | Bitwise equal to our reruns of 2026-11-07 in the same environment |
| Fine-tuning and scoring (step 3) | **Not rerun for this release**; they take days on four GPUs |
| Noise, structured-missing, feedback, pooling and label-headroom analyses, Fig. 5 data | **Not rerun with this release's code**; on MELD and M3ED they may stop at a consistency check against the 2026-10-29 files (see below) |
| A third party starting from the official datasets | **Not tested** |

## Known reproduction differences

**Which files the paper's numbers come from.** The table scripts name each file: IEMOCAP uses the `*_fin_20261107_*` files, MELD and M3ED use the `*_20261029_*` files. The p values for the Holm correction come from the `*_pv_20261107_*` reruns.

**IEMOCAP** reruns equal the paper's files exactly.

**MELD and M3ED** reruns differ slightly from the 2026-10-29 files in the rows that use CCT and in a few control rows. The rows of MACD, MACD+GCT and the base model are equal. The largest absolute differences, in percentage points over all MELD and M3ED result files:

| Quantity | Largest difference |
|---|---|
| Mean over missing rates, any row | 0.070 |
| Paired difference between methods | 0.066 |
| End of a 95% interval | 0.094 |

For EmoCODA itself:

| Setting | WAR, UAR, MF1 in the paper's file | WAR, UAR, MF1 rerun |
|---|---|---|
| MELD | 63.486, 40.078, 43.427 | 63.481, 40.067, 43.419 |
| M3ED | 54.709, 41.890, 40.828 | 54.726, 41.938, 40.884 |

These differences do not change the significance of any comparison at the 95% level, over all 26 result groups. The reruns of 2026-11-07 and 2026-11-09 are bitwise equal to each other. **We have not found the cause.** The code, the data and the library versions were ruled out, and the differences first appeared between 2026-10-29 and 2026-10-31. So your numbers may match the reruns and not the 2026-10-29 files.

Because of this, the release records this check in the output's `pv_info` field on MELD and M3ED instead of stopping. That is the only change to the logic of the core and MatchHIST scripts. The other edits for this release replace server paths with environment variables and remove table functions and names unrelated to the paper.

## License

The code and the result files in this repository are released under the [Apache License 2.0](LICENSE). The license does not cover the datasets (IEMOCAP, MELD, M3ED) or the model weights; obtain them from their owners under their own licenses.
