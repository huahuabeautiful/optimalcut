import rasterio
import os
import geopandas as gpd

from config import *
from shp_processor import read_shp, count_targets, merge_close_targets
from mask_generator import generate_mask
from slice_calculator import calculate_slice_positions
from slice_optimizer import optimize_slice_positions
from slice_generator import generate_slices, create_output_dirs

# ❌ 已删除：from dataset_splitter import split_dataset_prevent_leakage

def validate_image_file(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ 找不到影像文件：{image_path}")
    ext = os.path.splitext(image_path)[1].lower()
    if ext == '.dat':
        hdr_path = os.path.splitext(image_path)[0] + '.hdr'
        if not os.path.exists(hdr_path):
            raise FileNotFoundError(f"❌ 缺失 ENVI 头文件：{os.path.basename(hdr_path)}")
        print(f"✅ 检测到 .dat 影像及其头文件。")
    elif ext in ['.tif', '.tiff']:
        print(f"✅ 检测到 GeoTIFF 格式影像。")

def main():
    try:
        print("步骤0：环境与文件依赖检查...")
        validate_image_file(INPUT_IMAGE_PATH)
        with rasterio.open(INPUT_IMAGE_PATH) as src:
            image_crs = src.crs
            image_width = src.width
            image_height = src.height
            print(f"  -> 获取到目标影像投影：{image_crs}")

        print("步骤1：读取SHP文件并对齐坐标系...")
        gdf = read_shp(INPUT_SHP_PATH)

        if gdf.crs is None:
            print("  -> ⚠️ SHP缺失投影，强制设置为影像投影...")
            gdf.set_crs(image_crs, allow_override=True, inplace=True)
        elif gdf.crs != image_crs:
            print(f"  -> ⚠️ 投影不一致，正在自动重投影...")
            gdf = gdf.to_crs(image_crs)
        else:
            print("  -> ✅ 投影匹配成功。")

        # 扫描识别合并前目标数量，并应用自定义前缀的编号列
        original_count = count_targets(gdf)
        gdf['gully_id'] = [f"{GULLY_PREFIX}_Gully_{i + 1}" for i in range(original_count)]
        print(f"  -> 原始目标数量（合并前）：{original_count}，已应用前缀 [{GULLY_PREFIX}] 完成独立编号。")

        # 保存整理优化后带有编号列的原始 SHP 文件 (这对于后续规则网格对齐非常重要)
        os.makedirs(OUTPUT_ROOT, exist_ok=True)
        gdf.to_file(OUTPUT_ORIGINAL_SHP_WITH_ID, driver="ESRI Shapefile")
        print(f"  -> ✅ 已生成含独立编号的原始SHP：{OUTPUT_ORIGINAL_SHP_WITH_ID}")

        print("步骤2：合并近距离目标（用于优化切割算法）...")
        merged_gdf, merged_target_count = merge_close_targets(
            gdf, distance_threshold_pixel=DISTANCE_THRESHOLD,
            resolution=RESOLUTION, output_merged_shp=OUTPUT_MERGED_SHP
        )
        print(f"  -> 合并后目标数量：{merged_target_count}")

        print("步骤3：生成黑白掩码...")
        mask_array, image_transform, image_meta = generate_mask(
            image_path=INPUT_IMAGE_PATH, shp_gdf=merged_gdf,
            mask_foreground=MASK_FOREGROUND, mask_background=MASK_BACKGROUND
        )

        print("步骤4：提取骨干线网络...")
        small_positions, large_positions, slice_gdf, centerline_gdf = calculate_slice_positions(
            image_width=image_width, image_height=image_height,
            merged_gdf=merged_gdf, image_transform=image_transform, slice_size=SLICE_SIZE
        )

        if not centerline_gdf.empty:
            os.makedirs(OUTPUT_CENTERLINE_SHP, exist_ok=True)
            centerline_gdf.to_file(os.path.join(OUTPUT_CENTERLINE_SHP, "centerlines.shp"), driver="ESRI Shapefile")

        print("步骤5 & 6：计算并优化切割框位置...")
        slice_positions, _ = optimize_slice_positions(
            small_positions=small_positions,
            large_positions=large_positions,
            centerline_gdf=centerline_gdf,
            image_transform=image_transform,
            image_width=image_width,
            image_height=image_height,
            slice_size=SLICE_SIZE,
            overlap_threshold=OPTIMIZE_OVERLAP_THRESHOLD
        )

        print("步骤7：生成切片并执行智能空间关联命名...")
        create_output_dirs(
            OUTPUT_IMAGE_SLICE, OUTPUT_MASK_SLICE, OUTPUT_SLICE_SHP,
            OUTPUT_MERGED_SHP, OUTPUT_GEO_INFO
        )
        generate_slices(
            image_path=INPUT_IMAGE_PATH, mask_array=mask_array,
            slice_positions=slice_positions, image_transform=image_transform,
            image_meta=image_meta, slice_size=SLICE_SIZE, resolution=RESOLUTION,
            output_image_slice=OUTPUT_IMAGE_SLICE, output_mask_slice=OUTPUT_MASK_SLICE,
            output_slice_shp=OUTPUT_SLICE_SHP, output_geo_info=OUTPUT_GEO_INFO,
            original_gdf=gdf
        )

        # ❌ 已删除：步骤8 的划分逻辑。现在 main.py 运行结束即代表智能切片全部准备就绪。
        print(f"\n🎉 智能切割任务圆满完成！切片存储于： {OUTPUT_ROOT}")
        print("💡 请继续执行规则网格切割脚本，最后使用 unified_dataset_splitter.py 进行同步划分。")

    except Exception as e:
        print(f"\n❌ 执行出错：{str(e)}")
        raise

if __name__ == "__main__":
    main()