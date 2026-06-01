import h3
import networkx as nx
import json
import numpy as np
import torch
import torch.nn as nn
from itertools import combinations

INITIAL_ZOOM = 4
SPLIT_THRESHOLD = 1.0
NODE2VEC_DIM = 64
ECO_DIM = 16
TOTAL_EMBED_DIM = NODE2VEC_DIM + ECO_DIM


def build_h3_graph(hourly_df, split_threshold=SPLIT_THRESHOLD):
    G = nx.DiGraph()
    zoom = INITIAL_ZOOM

    hourly_df = hourly_df.copy()
    hourly_df['h3_region'] = hourly_df.apply(
        lambda r: h3.latlng_to_cell(r['latitude'], r['longitude'], zoom),
        axis=1
    )

    regions = hourly_df['h3_region'].unique()
    final_regions = {}

    for region in regions:
        pts = hourly_df[hourly_df['h3_region'] == region][
            ['latitude', 'longitude']
        ].values

        if len(pts) < 2:
            final_regions[region] = zoom
            continue

        sampled = pts[:50]
        dists = []
        for a, b in combinations(sampled, 2):
            dists.append(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))
        diameter = max(dists) if dists else 0

        if diameter > split_threshold:
            finer_zoom = zoom + 2
            sub = hourly_df[hourly_df['h3_region'] == region]
            for _, row in sub.iterrows():
                finer = h3.latlng_to_cell(row['latitude'], row['longitude'], finer_zoom)
                final_regions[finer] = finer_zoom
        else:
            final_regions[region] = zoom

    def get_final_h3(lat, lon):
        coarse = h3.latlng_to_cell(lat, lon, zoom)
        z = final_regions.get(coarse, zoom)
        return h3.latlng_to_cell(lat, lon, z)

    hourly_df['h3_final'] = hourly_df.apply(
        lambda r: get_final_h3(r['latitude'], r['longitude']),
        axis=1
    )

    for region in hourly_df['h3_final'].unique():
        pts = hourly_df[hourly_df['h3_final'] == region]
        node_attrs = {}
        for col, attr in [
            ('NDVI', 'ndvi_mean'), ('water_occ_1km', 'ndwi_mean'),
            ('EVI', 'evi_mean'), ('LST_celsius', 'lst_mean'),
            ('elevation_m', 'elev_mean')
        ]:
            node_attrs[attr] = float(pts[col].mean()) if col in pts.columns else 0.0
        if 'LULC_class' in pts.columns:
            node_attrs['lulc_mode'] = int(pts['LULC_class'].mode()[0])
        if 'season_code' in pts.columns:
            node_attrs['ndvi_dry'] = float(pts[pts['season_code'] == 0]['NDVI'].mean()) if len(pts[pts['season_code'] == 0]) > 0 else 0.0
            node_attrs['ndvi_wet'] = float(pts[pts['season_code'] == 1]['NDVI'].mean()) if len(pts[pts['season_code'] == 1]) > 0 else 0.0
            node_attrs['water_dry'] = float(pts[pts['season_code'] == 0]['water_occ_1km'].mean()) if 'water_occ_1km' in pts.columns and len(pts[pts['season_code'] == 0]) > 0 else 0.0
            node_attrs['water_wet'] = float(pts[pts['season_code'] == 1]['water_occ_1km'].mean()) if 'water_occ_1km' in pts.columns and len(pts[pts['season_code'] == 1]) > 0 else 0.0
        G.add_node(region, **node_attrs)

    if 'elephant_id' in hourly_df.columns:
        for eleph_id, group in hourly_df.groupby('elephant_id'):
            group = group.sort_values('timestamp')
            regions_seq = group['h3_final'].values
            for src, dst in zip(regions_seq[:-1], regions_seq[1:]):
                if src != dst:
                    if G.has_edge(src, dst):
                        G[src][dst]['count'] += 1
                    else:
                        G.add_edge(src, dst, count=1)

    return G, hourly_df


def train_node2vec(G, dimensions=NODE2VEC_DIM):
    from node2vec import Node2Vec
    node2vec = Node2Vec(
        G,
        dimensions=dimensions,
        walk_length=40,
        num_walks=200,
        p=1,
        q=4,
        workers=4,
        quiet=True
    )
    model = node2vec.fit(window=10, min_count=1)
    return model


def build_ecological_embeddings(G, node2vec_model):
    eco_proj = nn.Linear(4, ECO_DIM)

    embeddings = {}
    for node in G.nodes():
        structural = torch.tensor(node2vec_model.wv[node], dtype=torch.float32)
        eco_raw = torch.tensor([
            G.nodes[node].get('ndvi_mean', 0),
            G.nodes[node].get('evi_mean', 0),
            G.nodes[node].get('lst_mean', 0),
            G.nodes[node].get('ndwi_mean', 0),
        ], dtype=torch.float32)
        eco_emb = eco_proj(eco_raw)
        embeddings[node] = torch.cat([structural, eco_emb]).detach().numpy()

    return embeddings


def main():
    import argparse
    import os
    import pickle
    import pandas as pd

    parser = argparse.ArgumentParser(description="Build H3 graph from hourly resampled data")
    parser.add_argument("--input", type=str, required=True,
                        help="Directory containing hourly CSV files")
    parser.add_argument("--output", type=str, required=True,
                        help="Directory to write H3 graph artifacts")
    parser.add_argument("--split-threshold", type=float, default=SPLIT_THRESHOLD,
                        help=f"Diameter threshold for H3 region splitting (default: {SPLIT_THRESHOLD})")
    args = parser.parse_args()

    hourly_files = [f for f in os.listdir(args.input) if f.endswith('.csv')]
    if not hourly_files:
        raise FileNotFoundError(f"No CSV files found in {args.input}")

    dfs = []
    for f in hourly_files:
        df = pd.read_csv(os.path.join(args.input, f), parse_dates=['timestamp'])
        dfs.append(df)
    hourly_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(hourly_df)} hourly rows from {len(hourly_files)} files")

    G, _ = build_h3_graph(hourly_df, split_threshold=args.split_threshold)
    print(f"Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    os.makedirs(args.output, exist_ok=True)

    with open(os.path.join(args.output, 'graph.pkl'), 'wb') as f:
        pickle.dump(G, f)

    node2vec_model = train_node2vec(G)
    embeddings = build_ecological_embeddings(G, node2vec_model)

    np.save(os.path.join(args.output, 'node_embeddings.npy'), embeddings)

    node_features = {}
    for node in G.nodes():
        node_features[node] = {k: v for k, v in G.nodes[node].items()
                               if isinstance(v, (int, float, str))}
    with open(os.path.join(args.output, 'node_features.json'), 'w') as f:
        json.dump(node_features, f, default=float)

    latent_dict = {}
    for node in G.nodes():
        ndvi_dry = G.nodes[node].get('ndvi_dry', 0.0)
        ndvi_wet = G.nodes[node].get('ndvi_wet', 0.0)
        water_dry = G.nodes[node].get('water_dry', 0.0)
        water_wet = G.nodes[node].get('water_wet', 0.0)
        latent_dict[node] = {
            'dry':  {'ndvi': ndvi_dry,  'water': water_dry},
            'wet':  {'ndvi': ndvi_wet,  'water': water_wet},
        }
    with open(os.path.join(args.output, 'latent_dict.pkl'), 'wb') as f:
        pickle.dump(latent_dict, f)

    print(f"H3 graph artifacts saved to {args.output}/")


if __name__ == "__main__":
    import numpy as np
    main()
