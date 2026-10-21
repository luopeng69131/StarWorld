# Starlink Data Preparation

## Source and Attribution

The measurement source is **StarNet**, released by Connected Systems Lab:

- Official repository and regional download links: <https://github.com/ConnectedSystemsLab/StarNet>
- Dataset paper: Zikun Liu, Fan-Xue Gabriella Reidys, Sarah Tanveer, and Deepak Vasisht. *Vivisecting Starlink Throughput: Measurement and Prediction*. Proceedings of the ACM on Networking, 3(CoNEXT4), 2025, pp. 1–23. DOI: [10.1145/3768971](https://doi.org/10.1145/3768971).

Obtain the United States, Germany, and Canada measurements using the official download links. Preserve the attribution and data-use terms supplied by the original authors.

## Input Layout

The importer reads extracted `network_measurements.log` files and such files inside ZIP archives:

```text
starnet/
  data_4_26_us.zip
  data_osn.zip
  data_victoria.zip
```

An extracted run can instead appear as `data_osn/<run>/network_measurements.log`. The parser recognizes throughput records of the form:

```text
2025-01-01 00:00:00 TP: 80.0
2025-01-01 00:00:01 TP: 76.5
```

Throughput is in Mbps. Timestamps retain the acquisition clock. Region labels map to Chicago, United States (`US`), Osnabrück, Germany (`OSN`), and Victoria, Canada (`VIC`). For custom folder names, supply `--region US`, `--region OSN`, or `--region VIC`.

## Measurement Processing

```bash
starworld prepare-data \
  --input-root /path/to/starnet \
  --output outputs/tables
```

The pipeline:

1. Averages throughput observations in the same second and retains zero-capacity observations.
2. Interpolates interior gaps whose endpoint separation is at most two seconds by default; longer gaps start a new continuous block.
3. Creates complete, nonoverlapping 600-second segments within each block.
4. At each 10-second boundary, derives features from the preceding 60 seconds and stores the next 10-second statistics in separate target columns.
5. Produces stable content-derived collection-run and segment identifiers; repeated copies of the same log are processed once.
6. Records source names, hashes, processing parameters, and source attribution in `preparation.json`.

The bundled prediction adapter supplies a historical-mean point estimate and a historical 0.1-quantile safe-capacity estimate. Custom causal predictor outputs can replace the corresponding columns in the prediction table.

Generated files:

| File | Key fields |
| --- | --- |
| `segments.csv` | `trace_id`, `source_run_id`, `region`, `duration_s`, `start_time` |
| `prediction_windows.csv` | Window boundaries, history statistics, causal predictor outputs, and future targets |
| `replay_index.csv` | Measurement-to-playback log associations |
| `config.json` | Portable paths to the three tables |
| `preparation.json` | Provenance and preparation summary |

Paths in the generated configuration are relative to that configuration file. Dataset construction requires at least three independent collection runs to form train, validation, and test groups.

## Playback Records

Attach replay outputs using the generated segment identifiers. Populate `outputs/tables/replay_index.csv` with:

```csv
trace_id,controller,log_path
OSN_<run-digest>_segment_00000,BOLA,replays/bola.log
OSN_<run-digest>_segment_00000,RobustMPC,replays/robustmpc.log
```

Replay paths may be absolute or relative to the replay-index CSV. Each log records completed chunks, with columns:

| Position | Value | Unit |
| --- | --- | --- |
| 1 | Completion timestamp | Seconds |
| 2 | Bitrate | Kbps |
| 3 | Buffer after completion | Seconds |
| 4 | Chunk rebuffering | Seconds |
| 5 | Chunk size | Bytes |
| 6 | Download delay | Milliseconds |
| 7 | Reward | Controller reward units |

The parser also supports the 12-column SafeSABR-style format, whose reward is column 8 and intervention flag is column 11. Supported controller identifiers are `BOLA`, `RobustMPC`, `Pensieve`, and `SafeSABR`.

The default replay contract uses 4-second video chunks, 48 planned chunks, and a 120-Mbps bitrate cap. The configuration features encode these predeclared values. Use these settings when generating playback logs for the supplied constructor.

An existing index can also be attached during preparation:

```bash
starworld prepare-data \
  --input-root /path/to/starnet \
  --replay-index /path/to/replay_index.csv \
  --output outputs/tables
```

## Cross-Layer Alignment

```bash
starworld build-dataset \
  --config outputs/tables/config.json \
  --output outputs/dataset
```

The constructor aligns prediction windows with the most recently completed playback chunk. Its action record summarizes decisions within the following half-open 10-second interval. Pre-completion and terminal states carry explicit availability masks. In-flight download state is retained for analytical playback continuation.

Each measurement segment contributes one complete network/prediction episode. Indexed replay logs add application branches. Future measurement fields become observable only after their measurement window completes; at their original timestamp, they remain supervision targets. Full-segment profile information is kept as metadata.

Output consists of `episodes.npz`, `index.json`, and `audit.json`. A collection run, all its overlapping windows, and all its controller branches belong to the same partition. Replay-supervised branches supply the playback targets used in cross-layer training.

See [usage.md](usage.md) for training, validation calibration, and imagination queries.
