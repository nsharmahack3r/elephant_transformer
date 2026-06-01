import h3
import numpy as np


class EcologicalOccupancySampler:
    def __init__(self, fine_zoom=9):
        self.fine_zoom = fine_zoom

    def sample_point(self, h3_region, season, ndvi_raster=None,
                     water_raster=None):
        try:
            dots = list(h3.h3_to_children(h3_region, self.fine_zoom))
        except Exception:
            boundary = h3.h3_to_geo_boundary(h3_region)
            lats = [p[0] for p in boundary]
            lons = [p[1] for p in boundary]
            return (
                np.random.uniform(min(lons), max(lons)),
                np.random.uniform(min(lats), max(lats))
            )

        weights = []
        for dot in dots:
            lat, lon = h3.h3_to_geo(dot)
            ndvi = self._sample_raster(ndvi_raster, lat, lon) if ndvi_raster else 0.0
            water = self._sample_raster(water_raster, lat, lon) if water_raster else 0.0

            if season == 'dry':
                w = 0.3 * max(ndvi, 0) + 0.7 * max(water, 0)
            else:
                w = 0.6 * max(ndvi, 0) + 0.4 * max(water, 0)

            weights.append(max(w, 0.01))

        probs = np.array(weights) / sum(weights)
        chosen_dot = np.random.choice(dots, p=probs)

        boundary = h3.h3_to_geo_boundary(chosen_dot)
        lats = [p[0] for p in boundary]
        lons = [p[1] for p in boundary]
        lat = np.random.uniform(min(lats), max(lats))
        lon = np.random.uniform(min(lons), max(lons))

        return lon, lat

    def _sample_raster(self, raster, lat, lon):
        try:
            return float(raster.sample([(lon, lat)]).__next__()[0])
        except Exception:
            return 0.0

    def sample_route(self, region_sequence, season,
                     ndvi_raster=None, water_raster=None):
        points = []
        for region in region_sequence:
            lon, lat = self.sample_point(
                region, season, ndvi_raster, water_raster
            )
            points.append((lon, lat))
        return points
