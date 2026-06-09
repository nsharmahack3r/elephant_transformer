import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import pickle
import h3
import json
import os


def load_h3_zoom(h3_graph_dir):
    """
    Read the H3 zoom level written by h3_builder.py.
    Falls back to 6 if the file doesn't exist (safe default for existing graphs).
    """
    meta_path = os.path.join(h3_graph_dir, "h3_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)["zoom"]
    return 6  # default: matches INITIAL_ZOOM in h3_builder.py


class CoarseWindowDataset(Dataset):
    def __init__(self, hourly_csv_path, h3_graph_dir, window_size=24,
                 stride=12, split="train", train_frac=0.8, val_frac=0.1):
        """
        Args:
            split:      'train' | 'val' | 'test'
            train_frac: fraction of each elephant's timeline used for train
            val_frac:   fraction used for val; remainder is test
        """
        df = pd.read_csv(hourly_csv_path, parse_dates=["timestamp"])
        df = df.sort_values(["elephant_id", "timestamp"])

        zoom = load_h3_zoom(h3_graph_dir)
        df["h3_idx"] = df.apply(
            lambda r: h3.latlng_to_cell(r["latitude"], r["longitude"], zoom),
            axis=1,
        )

        all_h3 = sorted(df["h3_idx"].unique())
        self.h3_to_id = {h: i for i, h in enumerate(all_h3)}
        self.id_to_h3 = {i: h for h, i in self.h3_to_id.items()}
        self.num_h3_nodes = len(all_h3)

        embeddings_dict = np.load(
            os.path.join(h3_graph_dir, "node_embeddings.npy"), allow_pickle=True
        ).item()
        embed_dim = next(iter(embeddings_dict.values())).shape[0]

        self.node_embeddings = torch.zeros(self.num_h3_nodes, embed_dim)
        for h3_hex, emb in embeddings_dict.items():
            if h3_hex in self.h3_to_id:
                self.node_embeddings[self.h3_to_id[h3_hex]] = torch.tensor(
                    emb, dtype=torch.float32
                )

        df["node_id"] = df["h3_idx"].map(self.h3_to_id)

        self.samples = []
        for eleph_id, edf in df.groupby("elephant_id"):
            edf = edf.reset_index(drop=True)
            n = len(edf)

            # Temporal cut-points
            i_val  = int(n * train_frac)
            i_test = int(n * (train_frac + val_frac))

            if split == "train":
                edf = edf.iloc[:i_val]
            elif split == "val":
                edf = edf.iloc[i_val:i_test]
            else:  # test
                edf = edf.iloc[i_test:]

            if len(edf) < window_size:
                continue

            seq          = edf["node_id"].values.astype(np.int64)
            season_seq   = edf["season_code"].values.astype(np.int64)
            behavior_seq = edf["behavior_code"].values.astype(np.int64)

            occupancy_seq = np.ones(len(seq), dtype=np.float32)
            run_start = 0
            for j in range(1, len(seq)):
                if seq[j] != seq[j - 1]:
                    run_len = j - run_start
                    occupancy_seq[run_start:j] = float(run_len)
                    run_start = j
            occupancy_seq[run_start:] = float(len(seq) - run_start)
            occupancy_seq = occupancy_seq / window_size  # normalise to [0, 1]

            for start in range(0, len(seq) - window_size, stride):
                end = start + window_size
                self.samples.append({
                    "node_indices": seq[start:end - 1],
                    "target_nodes": seq[start + 1:end],
                    "occupancy": occupancy_seq[start:end - 1],
                    "season": int(np.bincount(season_seq[start:end]).argmax()),
                    "behavior": int(np.bincount(behavior_seq[start:end]).argmax()),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "node_indices": torch.tensor(s["node_indices"], dtype=torch.long),
            "target_nodes": torch.tensor(s["target_nodes"], dtype=torch.long),
            "occupancy": torch.tensor(s["occupancy"], dtype=torch.float32),
            "season": torch.tensor(s["season"], dtype=torch.long),
            "behavior": torch.tensor(s["behavior"], dtype=torch.long),
        }
