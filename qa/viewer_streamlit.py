import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st


@dataclass
class QAEntry:
    scene_id: str
    question_type: str
    sub_question_type: str
    question: str
    answer: str
    image_a: str
    image_a_mask: str
    image_b: str
    covisibility: float
    threshold: float
    camera_a: Dict
    camera_b: Dict
    raw: Dict
    # 大模型回答相关字段
    model_output: Optional[str] = None
    pred_answer: Optional[str] = None
    answer_gt: Optional[str] = None
    eval_question: Optional[str] = None  # 存储eval结果中的完整question，用于解析选项


@st.cache_data(show_spinner=False)
def load_eval_results(eval_dir: Path) -> Dict[str, List[Dict]]:
    """加载 output_eval 目录下的 jsonl 文件，按 scene_id 组织"""
    eval_results: Dict[str, List[Dict]] = {}
    if not eval_dir.exists():
        return eval_results
    
    # 查找所有 preds_vllm_{scene_id}_{timestamp}.jsonl 格式的文件
    for path in sorted(eval_dir.glob("preds_vllm_*.jsonl")):
        try:
            # 从文件名提取 scene_id: preds_vllm_{scene_id}_{timestamp}.jsonl
            parts = path.stem.split("_")
            if len(parts) >= 3 and parts[0] == "preds" and parts[1] == "vllm":
                scene_id = parts[2]
                results = []
                for line in path.read_text().splitlines():
                    if line.strip():
                        results.append(json.loads(line))
                if scene_id not in eval_results:
                    eval_results[scene_id] = []
                eval_results[scene_id].extend(results)
        except Exception as exc:
            st.warning(f"读取 eval 文件失败 {path.name}: {exc}")
            continue
    return eval_results


def match_eval_result(qa_entry: QAEntry, eval_results: List[Dict]) -> Optional[Dict]:
    """根据 question_type、image_a/image_b 匹配 eval 结果"""
    # 提取 image_a 和 image_b 的文件名（去掉路径）
    qa_image_a = Path(qa_entry.image_a).name if qa_entry.image_a else ""
    qa_image_b = Path(qa_entry.image_b).name if qa_entry.image_b else ""
    
    # 提取问题的核心部分（去掉选项部分，因为选项顺序可能不同）
    qa_question_base = qa_entry.question.split("\nA)")[0].strip() if "\nA)" in qa_entry.question else qa_entry.question.strip()
    
    for eval_item in eval_results:
        # 提取 eval 结果中的 image_a 和 image_b 文件名
        eval_image_a = Path(eval_item.get("image_a", "")).name if eval_item.get("image_a") else ""
        eval_image_b = Path(eval_item.get("image_b", "")).name if eval_item.get("image_b") else ""
        
        # 提取eval问题的核心部分
        eval_question = eval_item.get("question", "")
        eval_question_base = eval_question.split("\nA)")[0].strip() if "\nA)" in eval_question else eval_question.strip()
        
        # 匹配：question_type、问题核心部分、image_a/image_b
        if (eval_item.get("question_type", "") == qa_entry.question_type and
            eval_question_base == qa_question_base and
            eval_image_a == qa_image_a and
            eval_image_b == qa_image_b):
            return eval_item
    return None


def parse_options_from_question(question: str) -> Dict[str, str]:
    """从问题文本中解析选项，返回 {字母: 选项内容} 的字典"""
    options = {}
    if "\nA)" not in question:
        return options
    
    # 提取选项部分
    parts = question.split("\nA)")
    if len(parts) < 2:
        return options
    
    options_text = "A)" + parts[1]
    # 移除最后的提示文本
    if "Reply with only" in options_text:
        options_text = options_text.split("Reply with only")[0]
    
    # 解析每个选项
    import re
    pattern = r'([A-D]\))\s*([^\n]+)'
    matches = re.findall(pattern, options_text)
    for letter_option, content in matches:
        letter = letter_option[0]  # 提取字母 A, B, C, D
        options[letter] = content.strip()
    
    return options


@st.cache_data(show_spinner=False)
def load_entries(output_dir: Path, eval_dir: Optional[Path] = None) -> List[QAEntry]:
    entries: List[QAEntry] = []
    if not output_dir.exists():
        return entries
    
    # 加载 eval 结果
    eval_results_by_scene = {}
    if eval_dir:
        eval_results_by_scene = load_eval_results(eval_dir)
    
    for path in sorted(output_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            st.warning(f"读取失败 {path.name}: {exc}")
            continue
        if not isinstance(data, list):
            st.warning(f"文件格式非列表，跳过: {path.name}")
            continue
        for item in data:
            try:
                scene_id = item["scene_id"]
                entry = QAEntry(
                    scene_id=scene_id,
                    question_type=item.get("question_type", "unknown"),
                    sub_question_type=item.get("sub_question_type", "unknown"),
                    question=item.get("question", ""),
                    answer=item.get("answer", ""),
                    image_a=item.get("image_a", ""),
                    image_a_mask=item.get("image_a_mask", ""),
                    image_b=item.get("image_b", ""),
                    covisibility=float(item.get("covisibility", 0.0)),
                    threshold=float(item.get("threshold", 0.0)),
                    camera_a=item.get("camera_a", {}),
                    camera_b=item.get("camera_b", {}),
                    raw=item,
                )
                
                # 尝试匹配 eval 结果
                if scene_id in eval_results_by_scene:
                    matched_eval = match_eval_result(entry, eval_results_by_scene[scene_id])
                    if matched_eval:
                        entry.model_output = matched_eval.get("model_output")
                        entry.pred_answer = matched_eval.get("pred_answer")
                        entry.answer_gt = matched_eval.get("answer_gt")
                        entry.eval_question = matched_eval.get("question")
                
                entries.append(entry)
            except Exception as exc:
                st.warning(f"条目解析失败于 {path.name}: {exc}")
    return entries


def resolve_image_path(data_root: Path, scene_id: str, file_name: str) -> Path:
    return (
        data_root
        / scene_id
        / "dslr"
        / "resized_undistorted_images"
        / file_name
    )


def main():
    st.set_page_config(page_title="QA Viewer", layout="wide")
    st.title("3D QA Viewer (local)")

    with st.sidebar:
        st.header("路径配置")
        output_dir_str = st.text_input("output_QA 路径", value="output_QA")
        eval_dir_str = st.text_input("output_eval 路径", value="output_eval")
        data_root_str = st.text_input("scannetppv2/data 路径", value="scannetppv2/data")
        reload = st.button("重新加载")

    output_dir = Path(output_dir_str).expanduser().resolve()
    eval_dir = Path(eval_dir_str).expanduser().resolve() if eval_dir_str else None
    data_root = Path(data_root_str).expanduser().resolve()

    if reload:
        load_entries.clear()
        load_eval_results.clear()

    entries = load_entries(output_dir, eval_dir)
    if not entries:
        st.info("未找到任何条目，请确认 output_QA 目录及文件。")
        return

    scene_ids = sorted({e.scene_id for e in entries})
    qtypes = sorted({e.question_type for e in entries})
    sub_qtypes = sorted({e.sub_question_type for e in entries})

    with st.sidebar:
        st.header("过滤")
        sel_scenes = st.multiselect("选择 scene_id", scene_ids, default=scene_ids)
        sel_qtypes = st.multiselect("选择 question_type", qtypes, default=qtypes)
        sel_sub_qtypes = st.multiselect("选择 sub_question_type", sub_qtypes, default=sub_qtypes)

    filtered = [
        e
        for e in entries
        if e.scene_id in sel_scenes
        and e.question_type in sel_qtypes
        and e.sub_question_type in sel_sub_qtypes
    ]
    st.write(f"共 {len(entries)} 条，过滤后 {len(filtered)} 条。")

    if not filtered:
        st.warning("过滤后无结果。")
        return

    # 展示单条
    idx = st.slider("选择条目索引", 0, len(filtered) - 1, 0)
    e = filtered[idx]

    st.subheader(f"[{e.question_type}/{e.sub_question_type}] {e.scene_id} | idx {idx}")
    st.markdown(f"**Question:** {e.question}")
    
    # 处理选择题的答案显示
    if e.raw.get("options") and isinstance(e.raw["options"], dict):
        # 选择题：显示选项内容
        gt_option = e.raw["options"].get(e.answer, e.answer)
        st.markdown(f"**Answer (GT):** {e.answer}) {gt_option}")
    else:
        # 非选择题：直接显示答案
        st.markdown(f"**Answer (GT):** {e.answer}")
    
    # 显示大模型回答（如果存在）
    if e.pred_answer is not None:
        # 对于选择题，需要从eval结果中解析选项内容
        if e.raw.get("options") and isinstance(e.raw["options"], dict):
            # 选择题：比较答案字母
            gt_answer_letter = e.answer_gt if e.answer_gt else e.answer
            
            # 从eval结果的question中解析选项（因为选项顺序可能不同）
            eval_options = {}
            if e.eval_question:
                eval_options = parse_options_from_question(e.eval_question)
            
            # 获取预测答案的选项内容
            pred_option_content = ""
            if e.pred_answer in eval_options:
                pred_option_content = eval_options[e.pred_answer]
            elif e.pred_answer in e.raw["options"]:
                # 如果eval中没有，尝试从QA的options中获取（可能不准确）
                pred_option_content = e.raw["options"][e.pred_answer]
            
            # 判断是否正确：比较答案字母
            is_correct = e.pred_answer == gt_answer_letter
            correct_mark = "✅" if is_correct else "❌"
            
            if pred_option_content:
                st.markdown(f"**Model Answer:** {e.pred_answer}) {pred_option_content} {correct_mark}")
            else:
                st.markdown(f"**Model Answer:** {e.pred_answer} {correct_mark}")
        else:
            # 非选择题：直接比较答案
            gt_answer = e.answer_gt if e.answer_gt else e.answer
            is_correct = e.pred_answer == gt_answer
            correct_mark = "✅" if is_correct else "❌"
            st.markdown(f"**Model Answer:** {e.pred_answer} {correct_mark}")
        
        if e.model_output:
            with st.expander("Model Output (完整回答)"):
                st.text(e.model_output)
    else:
        st.info("未找到对应的大模型回答")
    
    st.markdown(
        f"covisibility: {e.covisibility:.3f} | threshold: {e.threshold}"
    )

    cols = st.columns(2)
    for col, title, fname, cam in [
        (cols[0], "Image A", e.image_a, e.camera_a),
        (cols[1], "Image B", e.image_b, e.camera_b),
    ]:
        with col:
            st.markdown(f"**{title}**: {fname}")
            img_path = resolve_image_path(data_root, e.scene_id, fname)
            if img_path.exists():
                st.image(str(img_path))
            else:
                st.error(f"找不到图片: {img_path}")
            st.expander("Intrinsics/Extrinsics").json(cam)

    # 仅展示用于提问的 image_a mask（如果存在且能找到文件）
    mask_path_str = e.raw.get("image_a_mask") or getattr(e, "image_a_mask", "")
    if mask_path_str:
        mask_path = Path(mask_path_str)
        if mask_path.exists():
            st.image(str(mask_path), caption="Mask (image_a_mask)")
        else:
            st.warning(f"image_a_mask 文件不存在: {mask_path}")
    mask_path_str = e.raw.get("image_b_mask") or getattr(e, "image_b_mask", "")
    if mask_path_str:
        mask_path = Path(mask_path_str)
        if mask_path.exists():
            st.image(str(mask_path), caption="Mask (image_b_mask)")
        else:
            st.warning(f"image_b_mask 文件不存在: {mask_path}")

    st.expander("原始条目 JSON").json(e.raw)


if __name__ == "__main__":
    main()

