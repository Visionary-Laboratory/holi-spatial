# PoseVLM QA格式详细总结 (output_QA)

本文档详细总结了output_QA目录中所有question_type + sub_question_type的字段结构、格式以及示例内容。

## 目录
1. [总体统计](#总体统计)
2. [通用字段](#通用字段)
3. [Question Types 详细分析](#question-types-详细分析)
4. [Marker Types 分析](#marker-types-分析)
5. [Hash生成策略讨论](#hash生成策略讨论)

## 总体统计

### Question Types 统计
- `cam_translation`: 270,443 条 (约41%)
- `cam_rotation`: 72,974 条 (约11%)
- `object_distance`: 520,497 条 (约79%)
- `object_relpos`: 406,858 条 (约62%)

### Sub Question Types 统计
- `main_dir`: 90,200 条 (仅用于cam_translation/cam_rotation)
- `distance_exact`: 90,121 条 (仅用于object_distance)
- `distance_threshold`: 90,122 条 (仅用于object_distance)
- `bbox_center_distance`: 307,513 条 (仅用于object_distance)
- `inter_object_distance`: 212,984 条 (仅用于object_distance)
- `relpos_A_facing_B`: 406,858 条 (仅用于object_relpos)
- 其他rotation相关sub_types: 各种组合 (pitch+yaw_dual_dominant等)

## 通用字段

所有QA条目都包含以下通用字段：

```json
{
  "scene_id": "00777c41d4",           // 场景ID
  "image_a": "DSC00883.JPG",          // 图像A文件名
  "image_b": "DSC00895.JPG",          // 图像B文件名
  "covisibility": 0.3051893711090088, // 共视度
  "threshold": 0.25,                  // 阈值
  "question_type": "...",             // 问题类型
  "sub_question_type": "...",         // 子问题类型
  "question": "...",                  // 问题文本
  "answer": "...",                    // 答案
  "camera_a": {...},                  // 相机A参数
  "camera_b": {...},                  // 相机B参数
  // 其他类型特定字段...
}
```

### Camera 格式
```json
{
  "intrinsics": {
    "fl_x": 464.03189647464455,
    "fl_y": 464.27202552772036,
    "cx": 876.0,
    "cy": 584.0,
    "w": 1752,
    "h": 1168,
    "camera_model": "PINHOLE"
  },
  "transform_matrix": [4x4数组]  // 相机变换矩阵
}
```

## Question Types 详细分析

### 1. cam_translation (相机平移)

#### 统计: 270,443 条
#### Sub Type: main_dir (90,200 条)

**特点**: 仅使用main_dir子类型，涉及相机运动方向判断。

**字段格式**:
```json
{
  // 通用字段...
  "question_type": "cam_translation",
  "sub_question_type": "main_dir",
  "question": "What is the primary camera motion direction from view A to view B in view A's coordinate?\nA) left-down\nB) right-up\nC) right-down\nD) left-up\nReply with only the option letter (A/B/C/D).",
  "answer": "C",
  "options": {
    "A": "left-down",
    "B": "right-up",
    "C": "right-down",
    "D": "left-up"
  },
  "translation_details": {
    "translation_world": [x, y, z],      // 世界坐标系下的平移向量
    "translation_world_norm": 2.4127,    // 平移向量模长
    "translation_cam": [x, y, z],        // 相机坐标系下的平移向量
    "translation_cam_norm": 2.4127,      // 相机坐标系下模长
    "dominant_axis": "x",                // 主轴 ("x", "y", "z")
    "primary_direction": "right",        // 主方向
    "direction_label": "right-forward-down", // 完整方向标签
    "covisibility": 0.3977              // 共视度
  }
}
```

**marker_type**: 无 (使用原图)

### 2. cam_rotation (相机旋转)

#### 统计: 72,974 条
#### Sub Types: 多种组合

**特点**: 涉及相机旋转角度判断，有多种子类型表示旋转轴组合。

**Sub Types 列表**:
- `yaw+pitch_single_dominant`: 31,507 条
- `pitch+yaw_dual_dominant`: 3,599 条
- `yaw+roll_single_dominant`: 24,899 条
- `pitch+roll_single_dominant`: 3,338 条
- `roll+pitch_single_dominant`: 13 条
- `yaw+pitch_dual_dominant`: 5,900 条
- `pitch+yaw_single_dominant`: 3,488 条
- `roll+yaw_dual_dominant`: 124 条
- `yaw+roll_dual_dominant`: 1,968 条
- `pitch+roll_dual_dominant`: 36 条
- `roll+pitch_dual_dominant`: 13 条

**字段格式** (以yaw+pitch_single_dominant为例):
```json
{
  // 通用字段...
  "question_type": "cam_rotation",
  "sub_question_type": "yaw+pitch_single_dominant",
  "question": "What is the camera rotation from view A to view B?\nA) rotate up and left\nB) rotate down and right\nC) rotate up and right\nD) rotate down and left\nReply with only the option letter (A/B/C/D).",
  "answer": "B",
  "options": {
    "A": "rotate up and left",
    "B": "rotate down and right",
    "C": "rotate up and right",
    "D": "rotate down and left"
  },
  "rotation_details": {
    "axis_pair": "yaw+pitch",           // 旋转轴对
    "axis_drop": "roll",                // 被忽略的轴
    "mode": "single_dominant",          // 模式
    "a1_deg": 68.14245553959435,       // 轴1角度(度)
    "a2_deg": 14.47230538030126,       // 轴2角度(度)
    "ratio_a2_a1": 0.21238309164951805, // 轴2/轴1比率
    "difficulty": "easy",               // 难度级别
    "rules": {
      "min_deg": 5.0,                   // 最小角度阈值
      "dual_ratio": 0.5,                // 双轴比率阈值
      "ratio_buffer": 0.1,              // 比率缓冲区
      "dual_min_deg": 5.0               // 双轴最小角度
    }
  }
}
```

**marker_type**: 无 (使用原图)

### 3. object_distance (物体距离)

#### 统计: 520,497 条
#### Sub Types:
- `bbox_center_distance`: 307,513 条
- `distance_exact`: 90,121 条
- `distance_threshold`: 90,122 条
- `inter_object_distance`: 212,984 条

**特点**: 涉及物体距离测量，不同子类型有不同的距离计算方式。

#### 3.1 bbox_center_distance
**字段格式**:
```json
{
  // 通用字段...
  "question_type": "object_distance",
  "sub_question_type": "bbox_center_distance",
  "question": "In image A, a separate white mask image highlights hose.\nLocate the same physical object in image B.\nEstimate the 3D metric distance (in meters)\nfrom the camera position of image B (camera center)\nto the \"hose\" (to the object surface/center point).\nThis is NOT pixel distance to the image center.\nReturn only one number in meters (e.g., 0.7).\nOutput format: <answer>NUMBER</answer>.",
  "answer": "2.4 m",
  "selected_ins_id": "9",                // 选中的实例ID
  "marker_type": "white_mask",          // 标记类型
  "marker_data": {},                    // 标记数据 (可为空)
  "objects": [                          // 物体列表
    {
      "ins_id": "9",
      "label": "hose",
      "mask_path_a": "/path/to/mask_a.png",
      "mask_path_b": "/path/to/mask_b.png",
      "bounding_box": [8个3D点坐标],    // 3D边界框
      "center_world": [x, y, z],        // 世界坐标系中心点
      "center_cam_b": [x, y, z],        // 相机B坐标系中心点
      "center_proj_b": null,            // 投影到图像B的中心点
      "distance_cam_b": 2.435215758041527 // 到相机B的距离
    }
  ]
}
```

#### 3.2 distance_exact
**特点**: 精确距离判断
```json
{
  // 类似bbox_center_distance...
  "sub_question_type": "distance_exact",
  "question": "What is the exact distance between the camera and the object?\nReturn format: X.XX m",
  "answer": "3.45 m"
}
```

#### 3.3 distance_threshold
**特点**: 阈值距离判断
```json
{
  // 类似bbox_center_distance...
  "sub_question_type": "distance_threshold",
  "question": "Is the object closer than X meters?\nA) Yes\nB) No",
  "answer": "A"
}
```

#### 3.4 inter_object_distance
**特点**: 物体间距离
```json
{
  // 包含两个物体...
  "sub_question_type": "inter_object_distance",
  "question": "What is the distance between object A and object B?",
  "objects": [
    {"ins_id": "1", "label": "chair", ...},
    {"ins_id": "2", "label": "table", ...}
  ]
}
```

### 4. object_relpos (物体相对位置)

#### 统计: 406,858 条
#### Sub Type: relpos_A_facing_B (406,858 条)

**特点**: 涉及物体间的相对位置关系，站在一个物体面向另一个物体的视角，判断第三个物体的方向。

**字段格式**:
```json
{
  // 通用字段...
  "question_type": "object_relpos",
  "sub_question_type": "relpos_A_facing_B",
  "question": "In image A, a red mask overlay indicates ladder.\nIn image B, a red mask overlay indicates ladder.\nIn image B, a green mask overlay indicates sign.\nIn image A, a blue mask overlay indicates box.\nYou are positioned at ladder and face sign.\nIn which direction is box relative to you?\nA. Front\nB. Left\nC. Back-Right\nD. Right",
  "answer": "D. (Right)",
  "marker_type": "mask_covering_image",  // 标记类型
  "marker_data": {                        // 标记数据
    "image_a_mask_rle_A": {               // 图像A的mask A (RLE格式)
      "size": [1168, 1752],
      "counts": "..."                     // RLE编码字符串
    },
    "image_a_mask_rle_A_color": "red",    // mask A颜色
    "image_b_mask_rle_A": {...},          // 图像B的mask A
    "image_b_mask_rle_A_color": "red",
    "image_b_mask_rle_B": {...},          // 图像B的mask B
    "image_b_mask_rle_B_color": "green",
    "image_a_mask_rle_C": {...},          // 图像A的mask C
    "image_a_mask_rle_C_color": "blue"
  },
  "mask_index_path": "/path/to/mask_index.json", // mask索引文件路径
  "objects": {                             // 物体信息 (A, B, C)
    "A": {
      "ins_id": "60",
      "label": "ladder",
      "center_world": [x, y, z]           // 世界坐标系中心点
    },
    "B": {
      "ins_id": "37",
      "label": "sign",
      "center_world": [x, y, z]
    },
    "C": {
      "ins_id": "55",
      "label": "box",
      "center_world": [x, y, z]
    }
  },
  "relpos_details": {                      // 相对位置详情
    "distance_m": 2.9943721294403076,     // 距离(米)
    "direction_label": "Right",           // 方向标签
    "direction_idx": 2,                   // 方向索引
    "yaw_deg": 90.00126539620214,         // 偏航角(度)
    "local_coords": {                     // 局部坐标系坐标
      "forward": -5.78761100769043e-05,
      "right": 2.620568037033081,
      "up": 1.448754072189331
    },
    "local_frame": {                      // 局部坐标系框架
      "forward": [x, y, z],               // 前向向量
      "right": [x, y, z],                 // 右向向量
      "up": [x, y, z]                     // 上向向量
    },
    "world_frame": {                      // 世界坐标系框架
      "forward": [x, y, z],
      "right": [x, y, z],
      "up": [x, y, z]
    }
  }
}
```

## Marker Types 分析

### 1. point_marker
**用途**: 在图像上标记点 (如object_dpt类型)
**数据格式**:
```json
"marker_data": {
  "image_a_point": [x, y],              // 点坐标
  "image_a_point_color": "red",         // 点颜色
  "image_b_point": [x, y],
  "image_b_point_color": "red"
}
```
**支持多个点**:
```json
"marker_data": {
  "image_a_point_1": [x1, y1],
  "image_a_point_1_color": "red",
  "image_a_point_2": [x2, y2],
  "image_a_point_2_color": "blue"
}
```

### 2. mask_covering_image
**用途**: 在原图上覆盖mask轮廓 (如object_relpos)
**数据格式**:
```json
"marker_data": {
  "image_a_mask_rle_A": {               // RLE格式mask
    "size": [height, width],
    "counts": "rle_string"
  },
  "image_a_mask_rle_A_color": "red",    // mask颜色
  "image_b_mask_rle_B": {...},
  "image_b_mask_rle_B_color": "green"
}
```

### 3. 2d_bbox
**用途**: 在图像上绘制2D边界框
**数据格式**:
```json
"marker_data": {
  "image_a_bbox": [x_min, y_min, x_max, y_max],
  "image_a_bbox_color": "red",
  "image_b_bbox": [x_min, y_min, x_max, y_max],
  "image_b_bbox_color": "blue"
}
```
**支持多个bbox**:
```json
"marker_data": {
  "image_a_bbox_A": [x_min, y_min, x_max, y_max],
  "image_a_bbox_A_color": "red",
  "image_a_bbox_B": [x_min, y_min, x_max, y_max],
  "image_a_bbox_B_color": "green"
}
```

### 4. white_mask
**用途**: 创建单独的白色mask图像 (如object_distance)
**数据格式**:
```json
"marker_data": {}  // 通常为空，mask信息在objects字段中
```
**实际mask数据存储在objects字段中**:
```json
"objects": [{
  "mask_path_a": "/path/to/mask.png",
  "mask_path_b": "/path/to/mask.png"
}]
```

### 5. 无标记 (language_description)
**用途**: 仅使用原图，不需要特殊标记
**数据格式**:
```json
"marker_type": null,
"marker_data": {}
```

## Hash生成策略讨论

### 当前Hash生成逻辑 (generate_unique_suffix函数)

当前hash基于以下信息生成:
1. `marker_type`
2. `question_type`
3. `sub_question_type`
4. 图片相关的marker_data信息 (点、mask、bbox坐标等)

### 用户提到的Hash改进需求

用户指出当前hash策略存在问题:

1. **不同指代类型必须有不同的hash**: ✓ (已实现)
2. **同一个图片的同一个指代类型区分**: 需要考虑`ins_id`的全局一致性
3. **颜色属性**: 即使ins_id集合相同，不同颜色的物体也应有不同hash

### 建议的Hash改进方案

**基础原则**:
- 不同`question_type` + `sub_question_type`组合必须有不同hash
- 相同类型但不同物体的QA必须有不同hash
- 相同物体但不同颜色/标记方式的QA必须有不同hash

**改进的Hash生成策略**:
```python
def generate_improved_unique_suffix(entry: Dict[str, Any], marker_data: Dict[str, Any], image_name: str) -> str:
    key_parts = []

    # 1. 基础类型信息
    question_type = entry.get("question_type", "")
    sub_question_type = entry.get("sub_question_type", "")
    marker_type = entry.get("marker_type", "")

    key_parts.extend([question_type, sub_question_type, marker_type])

    # 2. 物体身份信息 (ins_id)
    if "objects" in entry:
        if isinstance(entry["objects"], list):
            # object_distance格式
            ins_ids = sorted([obj.get("ins_id", "") for obj in entry["objects"]])
            key_parts.append("_".join(ins_ids))
        elif isinstance(entry["objects"], dict):
            # object_relpos格式 (A, B, C)
            ins_ids = []
            for obj_key in sorted(entry["objects"].keys()):
                obj = entry["objects"][obj_key]
                ins_ids.append(obj.get("ins_id", ""))
            key_parts.append("_".join(ins_ids))

    # 3. 标记颜色信息
    color_parts = []
    for key, value in marker_data.items():
        if key.endswith("_color"):
            color_parts.append(f"{key}:{value}")
    if color_parts:
        key_parts.append("_".join(sorted(color_parts)))

    # 4. 空间位置信息 (用于区分相同物体不同视图)
    if image_name == entry.get("image_a"):
        # image_a的相关位置信息
        pass  # 根据具体需求添加
    elif image_name == entry.get("image_b"):
        # image_b的相关位置信息
        pass

    # 生成hash
    key_str = "_".join(str(p) for p in key_parts)
    hash_obj = hashlib.md5(key_str.encode('utf-8'))
    hash_hex = hash_obj.hexdigest()[:8]

    return hash_hex
```

### Hash冲突风险分析

**当前潜在冲突**:
1. 相同marker_type + question_type + sub_question_type的条目可能产生相同hash
2. 未考虑ins_id的全局唯一性
3. 忽略了颜色属性的差异

**改进后的优势**:
1. 包含ins_id信息，确保相同物体的不同QA有不同hash
2. 包含颜色信息，确保相同物体不同颜色的QA有不同hash
3. 保持计算效率，同时增强唯一性

### 建议的验证方法

1. **统计分析**: 检查相同hash的条目是否确实应该相同
2. **跨场景验证**: 确保相同ins_id在不同场景中被正确区分
3. **颜色一致性**: 验证相同ins_id的物体颜色是否在场景中保持一致

---

*文档生成时间: 2026年1月18日*
*基于output_QA目录的实际数据分析*
