import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import Polygon
import os


def read_shp(shp_path):
    """读取SHP文件，返回GeoDataFrame（兼容Polygon/MultiPolygon）"""
    if not os.path.exists(shp_path):
        raise FileNotFoundError(f"SHP文件不存在：{shp_path}")
    gdf = gpd.read_file(shp_path)
    # 新增：打印几何类型，方便排查
    print("SHP文件中的几何类型：", gdf.geometry.type.unique())

    # 修复：筛选所有面要素（Polygon + MultiPolygon）
    valid_geom_types = ['Polygon', 'MultiPolygon']
    gdf = gdf[gdf.geometry.type.isin(valid_geom_types)]

    # 可选：将MultiPolygon拆解为单个Polygon（避免后续处理报错）
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)

    if len(gdf) == 0:
        raise ValueError("SHP文件中无有效面要素（Polygon/MultiPolygon）")
    return gdf

def count_targets(gdf):
    """统计SHP中的目标数量"""
    return len(gdf)


def merge_close_targets(gdf, distance_threshold_pixel, resolution, output_merged_shp):
    """
    合并距离≤阈值的目标（像素转地理距离：阈值(像素)*分辨率）
    :param gdf: 原始GeoDataFrame
    :param distance_threshold_pixel: 像素距离阈值
    :param resolution: 影像分辨率（m/像素）
    :param output_merged_shp: 合并后SHP的输出路径
    :return: 合并后的GeoDataFrame、合并后的目标数量
    """
    distance_threshold_m = distance_threshold_pixel * resolution
    merged_geoms = []
    remaining_geoms = list(gdf.geometry)

    while remaining_geoms:
        current = remaining_geoms.pop(0)
        # 寻找所有距离≤阈值的几何
        to_merge = [current]
        i = 0
        while i < len(remaining_geoms):
            if current.distance(remaining_geoms[i]) <= distance_threshold_m:
                to_merge.append(remaining_geoms.pop(i))
            else:
                i += 1
        # 合并几何
        merged_geom = unary_union(to_merge)
        merged_geoms.append(merged_geom)

    # 生成新的GeoDataFrame
    merged_gdf = gpd.GeoDataFrame(
        geometry=merged_geoms,
        crs=gdf.crs
    )
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_merged_shp), exist_ok=True)
    merged_gdf.to_file(output_merged_shp, driver="ESRI Shapefile")
    return merged_gdf, len(merged_geoms)


def get_geom_pixel_bounds(gdf, image_transform):
    """
    将地理坐标的几何转换为像素坐标的外接矩形
    :param gdf: 合并后的GeoDataFrame
    :param image_transform: 影像的转换矩阵（rasterio）
    :return: 列表，每个元素为(MinX, MinY, MaxX, MaxY)（像素坐标）
    """
    pixel_bounds = []
    for geom in gdf.geometry:
        # 地理坐标转像素坐标
        min_x, min_y, max_x, max_y = geom.bounds
        # 转换为像素坐标（rasterio的transform：x=col*a + row*b + x0；y=col*c + row*d + y0）
        min_col, min_row = ~image_transform * (min_x, min_y)
        max_col, max_row = ~image_transform * (max_x, max_y)
        # 取整并修正行列顺序（影像行从上到下）
        min_col, max_col = int(min(min_col, max_col)), int(max(min_col, max_col))
        min_row, max_row = int(min(min_row, max_row)), int(max(min_row, max_row))
        pixel_bounds.append((min_col, min_row, max_col, max_row))
    return pixel_bounds