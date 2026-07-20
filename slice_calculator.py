import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import affine_transform
import geopandas as gpd

from skeleton_extractor import extract_skeleton_from_polygon


def calculate_slice_positions(image_width, image_height, merged_gdf, image_transform, slice_size=256):
    small_positions = set()
    large_positions = set()
    slice_polygons = []
    centerline_geoms = []

    inv_transform = ~image_transform
    px_matrix = [inv_transform.a, inv_transform.b, inv_transform.d, inv_transform.e, inv_transform.c, inv_transform.f]
    geo_matrix = [image_transform.a, image_transform.b, image_transform.d, image_transform.e, image_transform.c,
                  image_transform.f]

    for geom_geo in merged_gdf.geometry:
        geom_px = affine_transform(geom_geo, px_matrix)

        min_col, min_row, max_col, max_row = geom_px.bounds
        target_width = max_col - min_col
        target_height = max_row - min_row

        # ==========================================
        # 规则1：小目标中心放置切割框，归入【锁死名单】
        # ==========================================
        if target_width <= slice_size and target_height <= slice_size:
            start_col = int((min_col + max_col) / 2 - slice_size / 2)
            start_row = int((min_row + max_row) / 2 - slice_size / 2)
            start_col = max(0, min(start_col, image_width - slice_size))
            start_row = max(0, min(start_row, image_height - slice_size))
            small_positions.add((start_col, start_row))
            continue

        # 大目标提取骨架
        lines = extract_skeleton_from_polygon(geom_px)

        for line_px in lines:
            line_geo = affine_transform(line_px, geo_matrix)
            centerline_geoms.append(line_geo)

        # 降级：提不出骨架则用滑窗兜底
        if not lines:
            x_starts = np.arange(min_col, max_col, slice_size * 0.8)
            y_starts = np.arange(min_row, max_row, slice_size * 0.8)
            for x in x_starts:
                for y in y_starts:
                    start_col = max(0, min(int(x), image_width - slice_size))
                    start_row = max(0, min(int(y), image_height - slice_size))
                    large_positions.add((start_col, start_row))
            continue

        for line in lines:
            coords = list(line.coords)
            if len(coords) < 2:
                continue

            first_batch = []
            skip_next = False

            for i in range(len(coords) - 1):
                if skip_next:
                    skip_next = False
                    continue

                p1 = coords[i][:2]
                p2 = coords[i + 1][:2]
                dx = abs(p2[0] - p1[0])
                dy = abs(p2[1] - p1[1])

                if dx < slice_size and dy < slice_size:
                    midpoint = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
                    first_batch.append(midpoint)
                    skip_next = True
                else:
                    first_batch.append(p1)

            if not skip_next:
                first_batch.append(coords[-1][:2])

            if len(first_batch) == 1:
                center_x, center_y = first_batch[0]
                start_col = max(0, min(int(center_x - slice_size / 2), image_width - slice_size))
                start_row = max(0, min(int(center_y - slice_size / 2), image_height - slice_size))
                large_positions.add((start_col, start_row))
                continue

            final_centers = []
            for i in range(len(first_batch) - 1):
                p1 = np.array(first_batch[i])
                p2 = np.array(first_batch[i + 1])

                if p1[0] > p2[0]:
                    p1, p2 = p2, p1

                final_centers.append(tuple(p1))

                segment_vector = p2 - p1
                L = np.linalg.norm(segment_vector)
                if L == 0: continue

                cos_theta = segment_vector[0] / L
                sin_theta = segment_vector[1] / L
                max_trig = max(abs(cos_theta), abs(sin_theta))

                if max_trig > 0:
                    d_step = slice_size / max_trig  # 保持 0% 完美贴合步长
                    if L > d_step:
                        num_intervals = int(L // d_step) + 1
                        for j in range(1, num_intervals):
                            new_p = p1 + segment_vector * (j / num_intervals)
                            final_centers.append(tuple(new_p))

                final_centers.append(tuple(p2))

            for center_x, center_y in final_centers:
                start_col = int(center_x - slice_size / 2)
                start_row = int(center_y - slice_size / 2)
                start_col = max(0, min(start_col, image_width - slice_size))
                start_row = max(0, min(start_row, image_height - slice_size))
                large_positions.add((start_col, start_row))

    small_positions = list(small_positions)
    large_positions = list(large_positions)
    all_positions = list(set(small_positions + large_positions))

    for (start_col, start_row) in all_positions:
        polygon = Polygon([
            (start_col, start_row), (start_col + slice_size, start_row),
            (start_col + slice_size, start_row + slice_size), (start_col, start_row + slice_size),
            (start_col, start_row)
        ])
        slice_polygons.append(polygon)

    slice_gdf = gpd.GeoDataFrame(geometry=slice_polygons, crs=None)
    centerline_gdf = gpd.GeoDataFrame(geometry=centerline_geoms, crs=merged_gdf.crs)

    # 返回拆分后的名单
    return small_positions, large_positions, slice_gdf, centerline_gdf