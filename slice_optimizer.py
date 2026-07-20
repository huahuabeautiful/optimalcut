import geopandas as gpd
from shapely.geometry import Polygon, Point, MultiLineString
from shapely.ops import nearest_points
from shapely.affinity import affine_transform


def calculate_overlap_ratio(p1, p2, slice_size):
    """
    严格计算两个正方形切割框之间的重叠率：交集面积 / 单个框面积
    """
    x1, y1 = p1
    x2, y2 = p2
    intersect_w = max(0, min(x1 + slice_size, x2 + slice_size) - max(x1, x2))
    intersect_h = max(0, min(y1 + slice_size, y2 + slice_size) - max(y1, y2))
    intersection_area = intersect_w * intersect_h
    box_area = slice_size * slice_size
    return intersection_area / box_area


def optimize_slice_positions(small_positions, large_positions, centerline_gdf, image_transform, image_width,
                             image_height, slice_size=256, overlap_threshold=0.5):
    # 规则1：小目标锁死不参与合并，只处理 large_positions
    positions = list(large_positions)

    # 准备骨干线的像素坐标 (用于中心点强制吸附)
    skeleton_multiline = None
    pixel_lines = []
    if not centerline_gdf.empty:
        inv_transform = ~image_transform
        px_matrix = [inv_transform.a, inv_transform.b, inv_transform.d, inv_transform.e, inv_transform.c,
                     inv_transform.f]
        for geom in centerline_gdf.geometry:
            if geom is not None and not geom.is_empty:
                pixel_lines.append(affine_transform(geom, px_matrix))
        if pixel_lines:
            skeleton_multiline = MultiLineString(pixel_lines)

    # ==========================================
    # 终极优化：成对独立计算 + 多轮迭代机制
    # ==========================================
    iteration_round = 0
    while True:
        iteration_round += 1
        merged_in_this_round = False

        # 每一轮从左往右排序，保证优化的方向性
        positions.sort(key=lambda p: (p[0], p[1]))

        n = len(positions)
        to_remove = set()  # 记录本轮已经被合并淘汰的框
        new_positions = []  # 记录本轮合并生成的新框

        for i in range(n):
            if i in to_remove:
                continue
            for j in range(i + 1, n):
                if j in to_remove:
                    continue

                # X轴跨度已经大于切片大小，绝对不可能重叠，提前跳出内循环提升性能
                if positions[j][0] - positions[i][0] >= slice_size:
                    break

                # 规则：仅计算两个独立切割框之间的重叠率
                if calculate_overlap_ratio(positions[i], positions[j], slice_size) > overlap_threshold:
                    # 取两个分割框的中心点
                    cx1 = positions[i][0] + slice_size / 2.0
                    cy1 = positions[i][1] + slice_size / 2.0
                    cx2 = positions[j][0] + slice_size / 2.0
                    cy2 = positions[j][1] + slice_size / 2.0

                    # 算两点之间的绝对中点
                    mid_cx = (cx1 + cx2) / 2.0
                    mid_cy = (cy1 + cy2) / 2.0

                    # 拓扑吸附：将这个悬空的中点，强制拉回骨干线上
                    if skeleton_multiline is not None:
                        mid_point = Point(mid_cx, mid_cy)
                        nearest_pt = nearest_points(skeleton_multiline, mid_point)[0]
                        final_cx, final_cy = nearest_pt.x, nearest_pt.y
                    else:
                        final_cx, final_cy = mid_cx, mid_cy

                    new_x = int(final_cx - slice_size / 2.0)
                    new_y = int(final_cy - slice_size / 2.0)
                    new_x = max(0, min(new_x, image_width - slice_size))
                    new_y = max(0, min(new_y, image_height - slice_size))

                    # 将生成的新框放入下一轮的候选池
                    new_positions.append((new_x, new_y))

                    # 标记这两个框在本轮已被消耗，不再参与本轮的其他计算
                    to_remove.add(i)
                    to_remove.add(j)
                    merged_in_this_round = True
                    break  # 发生合并后跳出内层寻找下一个未被消耗的框

        # 如果本轮没有任何一对框发生合并，说明全图重叠率均已达标，完美跳出多轮迭代
        if not merged_in_this_round:
            break

        # 结算本轮结果：保留未被淘汰的旧框，加入新融合的框，进入下一轮
        updated_positions = [positions[idx] for idx in range(n) if idx not in to_remove]
        updated_positions.extend(new_positions)
        positions = updated_positions

    print(f"      [系统提示] 50%成对重叠优化完毕，共进行了 {iteration_round} 轮迭代清扫。")

    # 将锁死的小目标与多轮迭代优化后的大目标合并
    final_positions = list(set(small_positions + positions))

    # ==========================================
    # 覆盖率回溯审查与动态打补丁 (保持不变)
    # ==========================================
    if pixel_lines:
        sample_points = []
        for line in pixel_lines:
            num_samples = int(line.length // 20) + 1
            for i in range(num_samples + 1):
                pt = line.interpolate(i / num_samples, normalized=True)
                sample_points.append((pt.x, pt.y))

        def is_covered(x, y, boxes):
            for bx, by in boxes:
                if bx <= x <= bx + slice_size and by <= y <= by + slice_size:
                    return True
            return False

        uncovered = [pt for pt in sample_points if not is_covered(pt[0], pt[1], final_positions)]

        supplement_count = 0
        while uncovered:
            pt_x, pt_y = uncovered[0]
            new_x = int(pt_x - slice_size / 2.0)
            new_y = int(pt_y - slice_size / 2.0)
            new_x = max(0, min(new_x, image_width - slice_size))
            new_y = max(0, min(new_y, image_height - slice_size))

            final_positions.append((new_x, new_y))
            supplement_count += 1

            uncovered = [pt for pt in uncovered if not is_covered(pt[0], pt[1], [(new_x, new_y)])]

        if supplement_count > 0:
            print(
                f"      [系统提示] 检测到合并操作导致目标边缘覆盖不足，已自动定位补充了 {supplement_count} 个框以确保100%覆盖。")

    # 去重并生成 GDF
    final_positions = list(set(final_positions))
    slice_polygons = []
    for (start_col, start_row) in final_positions:
        polygon = Polygon([
            (start_col, start_row), (start_col + slice_size, start_row),
            (start_col + slice_size, start_row + slice_size), (start_col, start_row + slice_size),
            (start_col, start_row)
        ])
        slice_polygons.append(polygon)

    slice_gdf = gpd.GeoDataFrame(geometry=slice_polygons, crs=None)

    return final_positions, slice_gdf