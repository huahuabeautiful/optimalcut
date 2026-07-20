import numpy as np
from shapely.geometry import LineString
from shapely.affinity import translate
from rasterio.features import rasterize
from scipy.spatial import cKDTree

try:
    from skimage.morphology import skeletonize
except ImportError:
    raise ImportError("请先安装 scikit-image 库：pip install scikit-image")


def extract_skeleton_from_polygon(polygon_px):
    """
    核心规则：面要素转像素 -> 向内收缩变线 -> 提取像素折线
    """
    minx, miny, maxx, maxy = polygon_px.bounds

    # 稍微向外扩展一点边界框，防止目标贴边
    pad = 2
    minx, miny = int(minx) - pad, int(miny) - pad
    maxx, maxy = int(maxx) + pad, int(maxy) + pad
    width, height = maxx - minx, maxy - miny

    if width <= 0 or height <= 0:
        return []

    # 第一步：面要素变成像素 (Rasterize)
    # 将面要素平移到局部坐标系 (0,0) 以便生成像素矩阵
    shifted_poly = translate(polygon_px, xoff=-minx, yoff=-miny)
    mask = rasterize([(shifted_poly, 1)], out_shape=(height, width), fill=0, dtype=np.uint8)

    # 第二步：向内收缩变成线要素 (Morphological Skeletonization)
    # skimage 的 skeletonize 会像剥洋葱一样从四周向内腐蚀，直到剩下1像素宽的绝对中心线
    skeleton = skeletonize(mask > 0)

    # 第三步：由线要素变成像素折线 (KDTree Vectorization)
    y, x = np.where(skeleton)
    if len(x) == 0:
        return []

    pts = np.column_stack((x, y))
    tree = cKDTree(pts)
    lines = []
    unvisited = set(range(len(pts)))

    # 沿着像素点顺藤摸瓜，串联成折线
    while unvisited:
        start_idx = unvisited.pop()
        path = [start_idx]

        # 向前搜索相连的像素点（距离 <= 1.5 涵盖了对角线相连的像素）
        curr = start_idx
        while True:
            dists, idxs = tree.query(pts[curr], k=9, distance_upper_bound=1.5)
            neighbors = [i for d, i in zip(dists, idxs) if d > 0 and d <= 1.5 and i in unvisited]
            if not neighbors: break
            next_node = neighbors[0]
            unvisited.remove(next_node)
            path.append(next_node)
            curr = next_node

        # 向后搜索相连的像素点
        curr = start_idx
        while True:
            dists, idxs = tree.query(pts[curr], k=9, distance_upper_bound=1.5)
            neighbors = [i for d, i in zip(dists, idxs) if d > 0 and d <= 1.5 and i in unvisited]
            if not neighbors: break
            next_node = neighbors[0]
            unvisited.remove(next_node)
            path.insert(0, next_node)
            curr = next_node

        if len(path) > 1:
            # 将局部像素坐标还原为整幅影像的真实像素坐标
            line_coords = [(pts[i][0] + minx, pts[i][1] + miny) for i in path]
            line = LineString(line_coords)
            # 简化像素密集的折线，拉直锯齿
            simplified_line = line.simplify(3.0)
            if simplified_line.geom_type == 'LineString' and len(simplified_line.coords) >= 2:
                lines.append(simplified_line)

    return lines