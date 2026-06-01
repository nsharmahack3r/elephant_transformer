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
        lambda r: h3.geo_to_h3(r['latitude'], r['longitude'], zoom),
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
                finer = h3.geo_to_h3(row['latitude'], row['longitude'], finer_zoom)
                final_regions[finer] = finer_zoom
        else:
            final_regions[region] = zoom

    def get_final_h3(lat, lon):
        coarse = h3.geo_to_h3(lat, lon, zoom)
        z = final_regions.get(coarse, zoom)
        return h3.geo_to_h3(lat, lon, z)

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
