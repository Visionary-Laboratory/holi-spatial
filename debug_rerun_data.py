#!/usr/bin/env python3
"""
诊断脚本：检查 rerun 文件中的数据结构和路径
"""

import argparse
import json
from pathlib import Path
import rerun.dataframe as rr_df
import numpy as np


def try_common_paths(rec):
    """尝试常见的路径模式来发现数据"""
    print(f"\n{'='*60}")
    print("尝试常见路径模式...")
    print(f"{'='*60}")
    
    common_paths = [
        "/instances",
        "instances",
        "/",
        None,  # 尝试根路径
    ]
    
    found_paths = []
    
    for path in common_paths:
        for index_type in ["log_time", "log_tick"]:
            try:
                view = rec.view(index=index_type, contents=path)
                data = view.select()
                df = data.read_pandas()
                if not df.empty:
                    print(f"\n✓ 在路径 '{path}' (索引: {index_type}) 找到数据:")
                    print(f"  形状: {df.shape}, 列: {list(df.columns)}")
                    print(f"  前几行:")
                    print(df.head(3))
                    found_paths.append((path, index_type, df))
                else:
                    print(f"  路径 '{path}' (索引: {index_type}): DataFrame 为空")
            except Exception as e:
                print(f"  路径 '{path}' (索引: {index_type}): 错误 - {type(e).__name__}: {e}")
    
    return found_paths


def explore_entity(rec, entity_path: str):
    """探索特定 entity 的数据"""
    print(f"\n探索路径: {entity_path}")
    
    # 尝试不同的索引方式
    for index_type in ["log_time", "log_tick"]:
        try:
            view = rec.view(index=index_type, contents=entity_path)
            data = view.select()
            df = data.read_pandas()
            
            if not df.empty:
                print(f"  ✓ 使用索引 '{index_type}' 成功获取数据:")
                print(f"    DataFrame 形状: {df.shape}")
                print(f"    列名: {list(df.columns)}")
                print(f"    行数: {len(df)}")
                
                # 显示前几行
                # if len(df) > 0:
                #     print(f"    前 3 行数据:")
                #     print(df.head(3).to_string())
                
                # 检查是否有 positions 相关的列
                pos_cols = [c for c in df.columns if "position" in c.lower()]
                if pos_cols:
                    print(f"    找到位置相关列: {pos_cols}")
                    for col in pos_cols:
                        val = df[col].iloc[-1]
                        print(f"      列 '{col}' 的最后一个值:")
                        print(f"        类型: {type(val)}")
                        if isinstance(val, np.ndarray):
                            print(f"        形状: {val.shape}")
                            print(f"        数据类型: {val.dtype}")
                            if val.size > 0 and val.size <= 20:
                                print(f"        值: {val}")
                            elif val.size > 0:
                                print(f"        前 5 个值: {val[:5] if val.ndim == 1 else val[:5, :]}")
                        else:
                            print(f"        值: {val}")
                
                return df
            else:
                print(f"  ✗ 使用索引 '{index_type}': DataFrame 为空")
        except Exception as e:
            print(f"  ✗ 使用索引 '{index_type}' 失败: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"  ✗ 所有尝试都失败")
    return None


def test_specific_paths(rec, label: str, ins_id: str):
    """测试特定 label 和 ins_id 的多种路径格式"""
    print(f"\n{'='*60}")
    print(f"测试路径: label={label}, ins_id={ins_id} (类型: {type(ins_id).__name__})")
    print(f"{'='*60}")
    
    # 尝试不同的路径格式和 ins_id 格式
    test_paths = []
    
    # 原始 ins_id
    test_paths.extend([
        f"/instances/{label}/{ins_id}/points",
        f"instances/{label}/{ins_id}/points",
        f"/instances/{label}/{ins_id}",
        f"instances/{label}/{ins_id}",
    ])
    
    # 尝试将 ins_id 转换为整数（如果它是字符串数字）
    try:
        ins_id_int = int(ins_id)
        test_paths.extend([
            f"/instances/{label}/{ins_id_int}/points",
            f"instances/{label}/{ins_id_int}/points",
            f"/instances/{label}/{ins_id_int}",
            f"instances/{label}/{ins_id_int}",
        ])
    except (ValueError, TypeError):
        pass
    
    # 尝试其他格式
    test_paths.extend([
        f"/{label}/{ins_id}/points",
        f"{label}/{ins_id}/points",
        f"/instances/{label}/points",
        f"instances/{label}/points",
    ])
    
    for path in test_paths:
        print(f"\n尝试路径: {path}")
        df = explore_entity(rec, path)
        if df is not None and not df.empty:
            print(f"  ✓ 成功找到数据!")
            return df
    
    print(f"\n  ✗ 所有路径都未找到数据")
    return None


def main():
    parser = argparse.ArgumentParser(description="诊断 rerun 文件中的数据")
    parser.add_argument("--rrd-path", type=Path, required=True, help="rerun 文件路径")
    parser.add_argument("--instances-json", type=Path, help="instances JSON 文件路径（可选）")
    parser.add_argument("--list-all", action="store_true", help="列出所有 entity 路径")
    parser.add_argument("--test-label", type=str, help="测试特定 label")
    parser.add_argument("--test-ins-id", type=str, help="测试特定 ins_id")
    
    args = parser.parse_args()
    
    # 加载 rerun 文件
    print(f"加载 rerun 文件: {args.rrd_path}")
    try:
        archive = rr_df.load_archive(str(args.rrd_path))
        recordings = archive.all_recordings()
        print(f"找到 {len(recordings)} 个 recording")
        
        if len(recordings) == 0:
            print("错误: 没有找到任何 recording")
            return
        
        rec = recordings[0]
        print(f"使用第一个 recording")
    except Exception as e:
        print(f"错误: 无法加载 rerun 文件: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 尝试常见路径
    if args.list_all:
        print(f"\nRecording 对象: {rec}")
        print(f"Recording 类型: {type(rec)}")
        # 尝试查看 recording 有哪些属性
        try:
            print(f"Recording 属性: {dir(rec)}")
        except:
            pass
        
        found = try_common_paths(rec)
        if not found:
            print("\n未找到任何数据，可能需要检查路径格式")
            print("\n建议:")
            print("  1. 使用 --instances-json 测试具体的实例路径")
            print("  2. 使用 --test-label 和 --test-ins-id 测试特定路径")
    
    # 测试特定路径
    if args.test_label and args.test_ins_id:
        test_specific_paths(rec, args.test_label, args.test_ins_id)
    
    # 如果提供了 instances_json，测试其中的所有实例
    if args.instances_json and args.instances_json.exists():
        print(f"\n{'='*60}")
        print(f"从 instances JSON 文件测试所有实例...")
        print(f"{'='*60}")
        
        with args.instances_json.open("r", encoding="utf-8") as f:
            instances_json = json.load(f)
        
        print(f"找到 {len(instances_json)} 个实例")
        
        # 先统计一下 label 和 ins_id 的分布
        label_counts = {}
        for inst in instances_json:
            label = inst.get("label", "unknown")
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"\nLabel 分布 (前 10 个):")
        for label, count in sorted(label_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {label}: {count} 个实例")
        
        # 统计成功和失败的实例
        success_count = 0
        fail_count = 0
        success_examples = []
        fail_examples = []
        
        # 先测试前几个，找到成功的例子
        test_count = min(5, len(instances_json))
        print(f"\n测试前 {test_count} 个实例...")
        
        for idx, inst in enumerate(instances_json[:test_count]):
            label = inst.get("label", "unknown")
            ins_id = inst.get("ins_id", "unknown")
            
            if label == "unknown" or ins_id == "unknown":
                continue
            
            # 确保 ins_id 是字符串
            ins_id = str(ins_id)
            
            print(f"\n[{idx+1}/{test_count}] 测试: label={label}, ins_id={ins_id}")
            df = test_specific_paths(rec, label, ins_id)
            if df is not None and not df.empty:
                success_count += 1
                if len(success_examples) < 3:
                    success_examples.append((label, ins_id))
                print(f"  ✓ 成功!")
            else:
                fail_count += 1
                if len(fail_examples) < 3:
                    fail_examples.append((label, ins_id))
                print(f"  ✗ 失败")
        
        print(f"\n{'='*60}")
        print(f"测试结果统计:")
        print(f"  成功: {success_count}/{test_count}")
        print(f"  失败: {fail_count}/{test_count}")
        
        if success_examples:
            print(f"\n  成功的例子:")
            for label, ins_id in success_examples:
                print(f"    - label={label}, ins_id={ins_id}")
        
        if fail_examples:
            print(f"\n  失败的例子:")
            for label, ins_id in fail_examples:
                print(f"    - label={label}, ins_id={ins_id}")
        
        print(f"{'='*60}")
        
        print(f"\n{'='*60}")
        print(f"测试结果: 成功 {success_count}, 失败 {fail_count}")
        print(f"{'='*60}")
    
    # 如果没有指定任何操作，默认尝试常见路径
    if not args.list_all and not (args.test_label and args.test_ins_id) and not args.instances_json:
        print(f"\n默认检查: 尝试常见路径...")
        try_common_paths(rec)
        print(f"\n提示: 使用 --list-all 列出所有路径，或使用 --test-label 和 --test-ins-id 测试特定路径")
        print(f"      或使用 --instances-json 测试 JSON 文件中的所有实例")


if __name__ == "__main__":
    main()

