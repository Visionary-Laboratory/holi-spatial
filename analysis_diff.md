# 诊断脚本 vs 原始脚本差异分析

## 关键差异对比

### 1. ins_id 类型处理

**诊断脚本** (`debug_rerun_data.py`):
```python
# 第238行
ins_id = str(ins_id)  # 确保 ins_id 是字符串
```

**原始脚本** (`2d_iou_gyn_seperate_instance_scene_label.py`):
```python
# 第182行
ins_id = inst.get("ins_id", "unknown")  # 直接使用，没有类型转换
```

**可能的问题**：
- 如果 JSON 中 `ins_id` 是整数类型（虽然从 JSON 看是字符串），但在某些情况下可能被解析为整数
- 路径拼接时，整数和字符串可能产生不同的路径格式

### 2. 列名查找方式

**诊断脚本** (`explore_entity` 函数):
```python
# 第71行
pos_cols = [c for c in df.columns if "position" in c.lower()]
```
- 使用 `"position"` (单数)
- 使用 `.lower()` 进行大小写不敏感匹配

**原始脚本** (`get_instance_points` 函数):
```python
# 第141行
col = [c for c in df.columns if "positions" in c]
```
- 使用 `"positions"` (复数)
- 不使用 `.lower()`，大小写敏感

**实际列名格式**（从诊断脚本输出）：
```
'/instances/box/11/points:Points3D:positions'
```

**分析**：
- 列名包含 `"positions"`（复数），所以原始脚本应该能找到
- 但诊断脚本使用 `"position"` + `.lower()` 也能找到（因为 "positions".lower() 包含 "position"）

### 3. 索引类型尝试

**诊断脚本** (`explore_entity` 函数):
```python
# 第53行
for index_type in ["log_time", "log_tick"]:  # 尝试两种索引
```

**原始脚本** (`get_instance_points` 函数):
```python
# 第127行
view = rec.view(index="log_time", contents=entity_path)  # 只使用 log_time
```

**分析**：
- 诊断脚本会尝试两种索引，如果 `log_time` 失败会尝试 `log_tick`
- 原始脚本只尝试 `log_time`，如果失败就直接返回 None

### 4. 错误处理

**诊断脚本**:
- 详细的错误信息输出
- 尝试多种路径格式
- 显示 DataFrame 的详细信息

**原始脚本**:
- 简单的错误处理
- 只尝试两种路径（带/不带前导斜杠）
- 错误信息较少

## 最可能的问题原因

基于诊断脚本能成功而原始脚本失败，最可能的原因是：

1. **ins_id 类型问题**（最可能）：
   - JSON 中 `ins_id` 可能是字符串，但在某些情况下可能被解析为其他类型
   - 路径拼接时 `f"/instances/{label}/{ins_id}/points"` 如果 `ins_id` 不是字符串，可能产生意外的路径格式

2. **索引类型问题**（次可能）：
   - 某些数据可能使用 `log_tick` 而不是 `log_time`
   - 原始脚本只尝试 `log_time`，如果数据使用 `log_tick` 就会失败

3. **列名查找问题**（不太可能）：
   - 虽然列名包含 "positions"，但大小写可能有问题
   - 原始脚本使用大小写敏感的匹配

## 建议的修复方案

1. **确保 ins_id 是字符串**：
   ```python
   ins_id = str(ins_id)
   ```

2. **改进列名查找**：
   ```python
   col = [c for c in df.columns if "position" in c.lower()]
   ```

3. **尝试多种索引类型**：
   ```python
   for index_type in ["log_time", "log_tick"]:
       try:
           view = rec.view(index=index_type, contents=entity_path)
           # ...
       except:
           continue
   ```









