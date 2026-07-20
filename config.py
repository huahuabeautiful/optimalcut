# ====================== 基础参数 ======================
RESOLUTION = 2.0  # 影像分辨率（m/像素）
SLICE_SIZE = 256  # 切片大小（像素）
DISTANCE_THRESHOLD = 100  # 目标合并距离阈值（像素）
OPTIMIZE_OVERLAP_THRESHOLD = 0.2  # 筛选优化的重叠率控制开关

# 🌟 新增：侵蚀沟自定义命名前缀与数据集划分比例
GULLY_PREFIX = "GF6_"  # 生成编号的前缀，例如 "NJ" 会生成 "NJ_Gully_1"
DATASET_SPLIT_RATIO = [0.8, 0.1, 0.1]  # 训练集、验证集、测试集比例

# ====================== 输入路径 ======================
INPUT_IMAGE_PATH = r"G:\model_code\filetreatpython\GFdata\GF6_nenjiang\NJ_NND.dat"
INPUT_SHP_PATH = r"G:\model_code\filetreatpython\GFdata\GF6_nenjiang\mask_rebuild\GF6_mask.shp"

# ====================== 输出路径 ======================
OUTPUT_ROOT = r"G:\model_code\filetreatpython\GFdata\GF6_nenjiang\optimal_segment"

# 🌟 新增：带编号的原始SHP输出路径与最终数据集目录
OUTPUT_ORIGINAL_SHP_WITH_ID = f"{OUTPUT_ROOT}/original_gullies_with_ids.shp"
OUTPUT_DATASET = f"{OUTPUT_ROOT}/dataset"

OUTPUT_IMAGE_SLICE = f"{OUTPUT_ROOT}/image_slices"
OUTPUT_MASK_SLICE = f"{OUTPUT_ROOT}/mask_slices"
OUTPUT_SLICE_SHP = f"{OUTPUT_ROOT}/slice_shp"
OUTPUT_MERGED_SHP = f"{OUTPUT_ROOT}/merged_shp"
OUTPUT_GEO_INFO = f"{OUTPUT_ROOT}/geo_info"
OUTPUT_CENTERLINE_SHP = f"{OUTPUT_ROOT}/centerline_shp"

# ====================== 掩码参数 ======================
MASK_FOREGROUND = 255  # 目标区域（白）
MASK_BACKGROUND = 0    # 背景区域（黑）