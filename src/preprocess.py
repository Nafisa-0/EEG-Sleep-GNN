import os
import mne
import torch
import numpy as np

from config import (
    RAW_PATH, GRAPH_PATH, CHANNELS,
    FS, EPOCH_DURATION, MAX_GRAPHS
)
from graph_builder import build_graph
from utils import map_label, log, setup_logger

setup_logger()

# ─────────────────────────────────────────
#  HYPNOGRAM MATCHER
# ─────────────────────────────────────────
def find_hypnogram(psg_file, all_files):
    base   = psg_file.split("-PSG.edf")[0]
    prefix = base[:-1]   # strip last char (technician code varies)
    for f in all_files:
        if f.startswith(prefix) and "Hypnogram" in f:
            return f
    return None


# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────
def main():
    files     = sorted(os.listdir(RAW_PATH))
    psg_files = [f for f in files if f.endswith("-PSG.edf")]

    cap_msg = f"{MAX_GRAPHS}" if MAX_GRAPHS else "all"
    log(f"Found {len(psg_files)} PSG files. Building up to {cap_msg} graphs...\n")

    graph_id      = 0
    skipped_files = 0
    samples_per_epoch = FS * EPOCH_DURATION   # 3000

    for psg_file in psg_files:
        if MAX_GRAPHS and graph_id >= MAX_GRAPHS:
            break

        psg_path = os.path.join(RAW_PATH, psg_file)
        hyp_file = find_hypnogram(psg_file, files)

        if hyp_file is None:
            log(f"  [SKIP] No hypnogram for {psg_file}")
            skipped_files += 1
            continue

        hyp_path = os.path.join(RAW_PATH, hyp_file)
        log(f"Processing {psg_file}  |  graphs so far: {graph_id}")

        try:
            raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
        except Exception as e:
            log(f"  [ERROR] Cannot read {psg_file}: {e}")
            skipped_files += 1
            continue

        available = raw.ch_names
        channels  = [ch for ch in CHANNELS if ch in available]

        if len(channels) < 2:
            log(f"  [SKIP] Not enough usable channels in {psg_file}: {available}")
            skipped_files += 1
            continue

        raw.pick(channels)
        data = raw.get_data()

        try:
            annot = mne.read_annotations(hyp_path)
        except Exception as e:
            log(f"  [ERROR] Cannot read hypnogram {hyp_file}: {e}")
            skipped_files += 1
            continue

        raw.set_annotations(annot)
        events, event_id = mne.events_from_annotations(raw, verbose=False)

        for event in events:
            if MAX_GRAPHS and graph_id >= MAX_GRAPHS:
                break

            start = event[0]
            end   = start + samples_per_epoch

            if end > data.shape[1]:
                continue

            label_desc = list(event_id.keys())[
                list(event_id.values()).index(event[2])
            ]
            label = map_label(label_desc)
            if label is None:
                continue

            segment = data[:, start:end]

            graph = build_graph(segment, label)
            if graph is None:
                continue

            fname = os.path.join(GRAPH_PATH, f"graph_{graph_id:05d}.pt")
            torch.save(graph, fname)
            graph_id += 1

    log(f"\n{'='*50}")
    log(f"Total graphs saved : {graph_id}")
    log(f"Files skipped      : {skipped_files}")
    log(f"Save location      : {GRAPH_PATH}")
    log("Done ✅")


if __name__ == "__main__":
    main()