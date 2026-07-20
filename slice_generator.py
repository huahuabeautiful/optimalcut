import rasterio
import numpy as np
from PIL import Image
import os
import json
from shapely.geometry import Polygon
import geopandas as gpd
from rasterio.windows import Window  # 🌟 新增：导入 Window 模块


def create_output_dirs(output_image_slice, output_mask_slice, output_slice_shp, output_merged_shp, output_geo_info):
    dirs = [output_image_slice, output_mask_slice, output_slice_shp, output_merged_shp, output_geo_info]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)


def normalize_image(array):
    normalized_bands = []
    for band in array:
        band = np.nan_to_num(band)
        min_v, max_v = np.min(band), np.max(band)
        if max_v - min_v == 0:
            normalized_band = np.zeros_like(band, dtype=np.uint8)
        else:
            normalized_band = ((band - min_v) / (max_v - min_v) * 255).astype(np.uint8)
        normalized_bands.append(normalized_band)
    norm_array = np.stack(normalized_bands, axis=-1)
    if norm_array.shape[-1] > 3:
        norm_array = norm_array[:, :, :3]
    elif norm_array.shape[-1] == 1:
        norm_array = norm_array[:, :, 0]
    return norm_array


def generate_slices(
        image_path, mask_array, slice_positions, image_transform, image_meta,
        slice_size=256, resolution=2.0,
        output_image_slice="output/image_slices",
        output_mask_slice="output/mask_slices",
        output_slice_shp="output/slice_shp",
        output_geo_info="output/geo_info",
        original_gdf=None
):
    # 记录每条侵蚀沟在所有切片中出现的次数序号
    gully_appearance_map = {gid: 0 for gid in original_gdf['gully_id']} if original_gdf is not None else {}
    slice_gdf_list = []

    # 🌟 优化：保持文件打开状态，在循环中按需读取局部窗口
    with rasterio.open(image_path) as src:
        for idx, (start_col, start_row) in enumerate(slice_positions):
            end_col, end_row = start_col + slice_size, start_row + slice_size

            # 🌟 核心修复：定义只涵盖当前 256x256 区域的 Window
            window = Window(col_off=start_col, row_off=start_row, width=slice_size, height=slice_size)

            # 🌟 核心修复：只读取这个 Window 的影像数据，不再一次性读取全图
            image_slice = src.read(window=window)

            # 掩码数组由于只有 8-bit 单通道，大约只占 3GB，仍可以直接从内存截取
            mask_slice = mask_array[start_row:end_row, start_col:end_col]

            # 计算地理范围
            geo_x = float(image_transform.xoff + start_col * resolution)
            geo_y = float(image_transform.yoff - start_row * resolution)
            slice_poly = Polygon([
                (geo_x, geo_y),
                (geo_x + slice_size * resolution, geo_y),
                (geo_x + slice_size * resolution, geo_y - slice_size * resolution),
                (geo_x, geo_y - slice_size * resolution),
                (geo_x, geo_y)
            ])

            # 【命名逻辑】：通过地理范围回溯原始侵蚀沟编号
            final_name = f"slice_{idx}"
            if original_gdf is not None:
                # 空间过滤：仅针对该切片范围内的目标进行点名
                possible_hits = original_gdf.iloc[list(original_gdf.sindex.intersection(slice_poly.bounds))]
                precise_hits = possible_hits[possible_hits.intersects(slice_poly)]

                if not precise_hits.empty:
                    name_parts = []
                    for _, row in precise_hits.sort_index().iterrows():
                        gid = row['gully_id']
                        gully_appearance_map[gid] += 1
                        name_parts.append(f"{gid}_{gully_appearance_map[gid]}")
                    final_name = "&".join(name_parts)

            # 保存影像与掩码
            norm_img = normalize_image(image_slice)
            Image.fromarray(norm_img).save(f"{output_image_slice}/{final_name}.jpg", quality=95)
            Image.fromarray(mask_slice.astype(np.uint8)).save(f"{output_mask_slice}/{final_name}.png")

            # 保存地理信息 JSON
            geo_data = {
                "name": final_name,
                "px_range": [[int(start_col), int(start_row)], [int(end_col), int(end_row)]],
                "geo_origin": (geo_x, geo_y),
                "crs": image_meta["crs"].to_wkt()
            }
            with open(f"{output_geo_info}/{final_name}.json", "w") as f:
                json.dump(geo_data, f, indent=4)

            slice_gdf_list.append({"slice_id": final_name, "geometry": slice_poly})

            # 可选：打印进度条（因为切片可能比较多）
            if (idx + 1) % 100 == 0:
                print(f"      -> 已生成 {idx + 1} / {len(slice_positions)} 个切片...")

    # 保存切片位置索引 SHP
    gpd.GeoDataFrame(slice_gdf_list, crs=image_meta["crs"]).to_file(
        f"{output_slice_shp}/slice_positions.shp", driver="ESRI Shapefile"
    )