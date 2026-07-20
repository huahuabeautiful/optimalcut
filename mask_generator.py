import rasterio
from rasterio.features import rasterize
import numpy as np
import os


def generate_mask(image_path, shp_gdf, mask_foreground=255, mask_background=0, output_mask_path=None):
    """
    生成与影像分辨率/尺寸匹配的黑白掩码
    :param image_path: 遥感影像路径
    :param shp_gdf: 合并后的GeoDataFrame
    :param mask_foreground: 掩码前景值（目标）
    :param mask_background: 掩码背景值（非目标）
    :param output_mask_path: 掩码保存路径（可选）
    :return: 掩码数组（numpy）、影像转换矩阵、影像元数据
    """
    # 读取影像元数据
    with rasterio.open(image_path) as src:
        image_meta = src.meta.copy()
        image_transform = src.transform
        image_width = src.width
        image_height = src.height
        # 检查投影是否一致
        if not shp_gdf.crs == src.crs:
            raise ValueError(f"SHP投影({shp_gdf.crs})与影像投影({src.crs})不一致！")

    # 生成掩码（面内foreground，背景background）
    shapes = [(geom, mask_foreground) for geom in shp_gdf.geometry]
    mask = rasterize(
        shapes=shapes,
        out_shape=(image_height, image_width),
        transform=image_transform,
        fill=mask_background,
        dtype=np.uint8
    )

    # 保存掩码（可选）
    if output_mask_path:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
        mask_meta = image_meta.copy()
        mask_meta.update({
            "dtype": np.uint8,
            "count": 1,
            "nodata": None
        })
        with rasterio.open(output_mask_path, "w", **mask_meta) as dst:
            dst.write(mask, 1)

    return mask, image_transform, image_meta