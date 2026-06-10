import json
import re
import sys
import multiprocessing
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from geo.Auxiliary_function import (
    check_coordinates_distinct,
    convert_coordinates,
    convert_conditions,
    eval_derived_coordinate,
)
from geo.coordinate_engine_config import COORDINATE_ENGINE_TIMEOUT
from geo.Kernel_function import extract_and_modify
from geo.latex_pdf_open import get_latex_code, for_render_code, render_latex_to_pdf

# ================= 路径常量 =================
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_DIR / "output"))
JSON_DIR = PROJECT_DIR / "json"
PROMPT_DIR = PROJECT_DIR / "prompt"



# ================= 配置与初始化 =================
client = OpenAI(
    api_key=os.getenv("API_KEY", "YOUR_API_KEY"),
    base_url=os.getenv("BASE_URL", "https://api.deepseek.com"),
)

MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-v4-flash")

# ================= 核心工具函数 =================

def call_llm(system_prompt, user_content, temperature=0.0, response_format=None):
    try:
        kwargs = dict(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
            stream=False,
        )
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content.strip()
        if response_format == "json":
            return json.loads(text)
        return text
    except Exception as e:
        print(f"LLM 调用失败: {e}")
        return {} if response_format == "json" else ""

def analyze_geometry_context(text):
    system_prompt = (
        "你是一个几何专家。请分析给出的数学题，并以 JSON 格式输出以下信息：\n"
        "1. type: 图形类型(circle/triangle/quad/mixed)\n"
        "2. subtype: 四边形子类型，仅当 type 为 quad 时填写 "
        "(square/rectangle/rhombus/parallelogram/generic_quad)，否则为 null\n"
        "3. radius: 如果涉及圆，提取半径数值；如果不确定但有半径概念，设为 1.0；否则为 null\n"
        "4. suggestions: 针对该图形的基本顶点坐标系建议（例如：'设圆心O为(0,0)', '设A为(0,0)'）。"
        "只建议顶点/圆心等基础点的坐标，不要写中点或交点的推导坐标（如 a/2、b/2）。"
        "若为四边形，给出轴对齐坐标模板（如正方形 A=(0,0),B=(0,s),C=(s,s),D=(s,0)）。\n"
        "5. is_circle: 布尔值，是否包含圆"
    )

    analysis = call_llm(system_prompt, f"题目内容：{text}", response_format="json")
    if not analysis:
        r_match = re.search(r"半径[为是等于]?\s*([0-9.]+)", text)
        radius = float(r_match.group(1)) if r_match else (5.0 if "圆" in text or "⊙" in text else None)
        return "类型判断失败，请根据常识设置坐标", radius

    radius = analysis.get('radius')
    if analysis.get('is_circle') and radius is None:
        radius = 1.0

    parts = [f"类型：{analysis.get('type')}"]
    subtype = analysis.get('subtype')
    if subtype:
        parts.append(f"子类型：{subtype}")
    parts.append(f"建议：{analysis.get('suggestions')}")
    extra_info = "。".join(parts)
    return extra_info, radius

def _resolve_coord_value(value, variables):
    if isinstance(value, str):
        return float(variables.get(value, [value])[0])
    return float(value)


def _is_derived_coord(coords):
    return len(coords) == 1 and isinstance(coords[0], str) and coords[0].startswith("calc_")


def _coordinates_fully_resolved(coordinates, variables):
    if variables:
        return False
    for coords in coordinates.values():
        if _is_derived_coord(coords):
            continue
        if len(coords) < 2:
            return False
        x, y = coords[0], coords[1]
        if isinstance(x, str) or isinstance(y, str):
            return False
    return True


def _resolve_point_coords(coords, variables, coordinates):
    if _is_derived_coord(coords):
        return eval_derived_coordinate(coords[0], variables, coordinates)
    if len(coords) < 2:
        return None
    x, y = coords[0], coords[1]
    if isinstance(x, str) or isinstance(y, str):
        return _resolve_coord_value(x, variables), _resolve_coord_value(y, variables)
    return float(x), float(y)


def _build_fusion(coordinates, variables, scale=2.0):
    fusion = {}
    for point, coords in coordinates.items():
        resolved = _resolve_point_coords(coords, variables, coordinates)
        if resolved is None:
            continue
        fusion[point] = (resolved[0] * scale, resolved[1] * scale)
    return fusion

def run_extract_and_modify(generated_points, condition_code, variables, output_queue):
    try:
        res = extract_and_modify(generated_points, condition_code, variables)
        output_queue.put(res)
    except Exception as e:
        output_queue.put(e)



def process_geometry_task(item, generic_knowledge, output_dir=None, json_stem=None):
    if output_dir is None:
        output_dir = OUTPUT_DIR
    if json_stem:
        output_dir = Path(output_dir) / json_stem
    output_dir.mkdir(parents=True, exist_ok=True)
    text = item['subject']
    print(f"\n--- 正在处理题目 ID: {item['id']} ---")
    

    extra_info, radius = analyze_geometry_context(text)
    print(f"[感知结果] 半径: {radius}, 辅助建议: {extra_info}")

    instruct = (
        "根据数学几何题意和辅助信息，输出 JSON，包含以下两个字段：\n"
        "1. coordinates: 字典，键为点名称，值为 [x, y] 数组（已知用数字，未知用单个变量名）。"
        "变量名只能是简单标识符（如 a、b、m、n、ox、oy），禁止算术表达式（如 a/2、2*a、a+b）。"
        "由 midpoint、online_inside 等条件确定的点，用独立变量表示，不要写推导坐标。\n"
        "2. conditions: 字典，键为条件编号，值为 [函数名, 参数1, 参数2, ...] 数组\n"
        "3. 若题目明确点在某圆（⊙O）上，除角度/弧中点等条件外，还必须为每个圆上点添加 "
        "'dist': ['dist', 'O', '点名称', 'r']（半径用 r，与坐标中的 r 一致）\n"
        "4. 若题目出现平行四边形/矩形/正方形/菱形，必须输出该图形的完整约束清单"
        "（见知识库「四边形子类型约束清单」），即使题目主要讨论内部点 E、F 等也不能省略。\n"
        "5. 正方形 few-shot 示例：\n"
        "{\"coordinates\": {\"A\": [0, 0], \"B\": [0, \"s\"], \"C\": [\"s\", \"s\"], "
        "\"D\": [\"s\", 0], \"E\": [\"e\", \"s\"]}, "
        "\"conditions\": {\"c1\": [\"parallel\", \"A\", \"B\", \"D\", \"C\"], "
        "\"c2\": [\"parallel\", \"A\", \"D\", \"B\", \"C\"], "
        "\"c3\": [\"ortho\", \"A\", \"B\", \"B\", \"C\"], "
        "\"c4\": [\"equal_line\", \"A\", \"B\", \"B\", \"C\"], "
        "\"c5\": [\"online_inside\", \"E\", \"B\", \"C\"]}}\n"
        "圆示例：{\"coordinates\": {\"O\": [0, 0], \"A\": [\"r\", 0], \"B\": [\"a\", \"b\"]}, "
        "\"conditions\": {\"c1\": [\"dist\", \"O\", \"A\", \"r\"], \"c2\": [\"angle\", \"B\", \"A\", \"C\", 35]}}\n"
    )

    full_prompt = f"{generic_knowledge}\n\n当前题目辅助背景：{extra_info}"
    user_msg = f"题目:{text}\n任务:{instruct}"
    
    print(full_prompt)
    print(user_msg)
    result = call_llm(full_prompt, user_msg, response_format="json")
    print(result)

    if not result:
        print("LLM 解析失败")
        return

    points_info = result.get("coordinates", {})
    cond_info = result.get("conditions", {})

    generated_points, variables = convert_coordinates(points_info, radius=radius)
    generated_points, variables, calc_cond, condition_code = convert_conditions(
        text, variables, generated_points, cond_info, radius=radius
    )
    
    if not condition_code:
        if (
            _coordinates_fully_resolved(generated_points, variables)
            and check_coordinates_distinct(generated_points, variables)
        ):
            print("所有条件已满足，跳过数值求解")
            fusion = _build_fusion(generated_points, variables)
            render_geometry_pdf(text, fusion, str(output_dir / f"{item['id']}.pdf"))
            return
        print("条件方程生成失败")
        return


    output_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=run_extract_and_modify, 
        args=(generated_points, condition_code, variables, output_queue)
    )
    process.start()
    process.join(COORDINATE_ENGINE_TIMEOUT) 

    if process.is_alive():
        process.terminate()
        print("核心计算引擎求解超时")
        return

    try:
        result = output_queue.get_nowait()
        if isinstance(result, Exception): raise result
        modified_coordinates, extracted_variables, found = result
        
        if found:
            fusion = _build_fusion(modified_coordinates, extracted_variables)
            render_geometry_pdf(text, fusion, str(output_dir / f"{item['id']}.pdf"))
        else:
            print("未能找到符合条件的数值解")
            
    except Exception as e:
        print(f"处理过程出错: {e}")

def _has_tikz_body(processed_code):
    """Check if processed LaTeX code contains actual TikZ drawing content."""
    return "\\begin{tikzpicture}" in processed_code


def _generate_tikz_fallback(text, fusion):
    """Generate basic TikZ code from fusion dict when LLM fails."""
    import math

    lines = [
        "\\usepackage{tikz}",
        "\\usetikzlibrary{calc}",
        "\\begin{document}",
        "\\begin{tikzpicture}",
    ]

    # Define coordinates
    for name, (x, y) in fusion.items():
        lines.append(f"  \\coordinate ({name}) at ({x:.4f}, {y:.4f});")

    lines.append("")

    # Detect and draw circle if center O exists and points are equidistant
    if "O" in fusion and ("⊙" in text or "圆" in text):
        ox, oy = fusion["O"]
        radii = {}
        for name, (x, y) in fusion.items():
            if name == "O":
                continue
            r = math.hypot(x - ox, y - oy)
            if r > 0.01:
                radii[name] = r
        if radii:
            avg_r = sum(radii.values()) / len(radii)
            lines.append(f"  \\draw (O) circle ({avg_r:.4f});")

    # Extract line segments from text: "连接OA，OB，AC，BC" etc.
    connection_patterns = re.findall(r"连接((?:[A-Z]{2}[,，、]?)+)", text)
    drawn_segments = set()
    for group in connection_patterns:
        pairs = re.findall(r"([A-Z])([A-Z])", group)
        for a, b in pairs:
            if a in fusion and b in fusion:
                seg = tuple(sorted((a, b)))
                drawn_segments.add(seg)

    for a, b in drawn_segments:
        lines.append(f"  \\draw ({a}) -- ({b});")

    # Label points
    lines.append("")
    for name in fusion:
        lines.append(f"  \\fill ({name}) circle (1.5pt) node[anchor=south] {{${name}$}};")

    lines.append("\\end{tikzpicture}")
    lines.append("\\end{document}")
    return "\n".join(lines)


def render_geometry_pdf(text, fusion, output_path):
    system_prompt = (
        "你是一个 LaTeX 绘图专家。根据给定的点坐标绘制几何图形。\n"
        "要求：使用 tikz 宏包，\\coordinate 定义点，\\draw 连线，"
        "\\fill (P) circle (1.5pt) node[anchor=south] {P} 标注点。\n"
        "不要包含任何角度符号（°）。只输出 LaTeX 代码。"
    )
    user_content = f"题目：{text}\n坐标数据：{json.dumps(fusion)}"

    latex_raw = call_llm(system_prompt, user_content)
    final_code = None
    try:
        clean_code = get_latex_code(latex_raw)
        final_code = for_render_code(clean_code)
    except Exception as e:
        print(f"LLM LaTeX 解析失败: {e}")

    # Fallback: if LLM returned empty/invalid TikZ, generate programmatically
    if final_code is None or not _has_tikz_body(final_code):
        print("[render] LLM TikZ body is empty, using programmatic fallback")
        final_code = _generate_tikz_fallback(text, fusion)

    try:
        render_latex_to_pdf(final_code, output_path)
        print(f"成功导出 PDF: {output_path}")
    except Exception as e:
        print(f"PDF 渲染阶段失败: {e}")



def main(input_json_path):
    generic_knowledge = (PROMPT_DIR / "geometry_knowledge.dat").read_text(encoding='utf-8')
    input_path = Path(input_json_path)
    if not input_path.exists():
        print(f"错误：找不到文件 {input_json_path}")
        return

    data = json.loads(input_path.read_text(encoding='utf-8'))
    json_stem = input_path.stem
    print(f"开始自动化处理任务，共 {len(data)} 题...")
    for item in data:
        process_geometry_task(item, generic_knowledge, json_stem=json_stem)

if __name__ == "__main__":
    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else JSON_DIR / "circle.json"
    if not json_path.is_absolute():
        json_path = PROJECT_DIR / json_path
    question_id = int(sys.argv[2]) if len(sys.argv) > 2 else None

    generic_knowledge = (PROMPT_DIR / "geometry_knowledge.dat").read_text(encoding='utf-8')

    if not json_path.exists():
        print(f"错误：找不到文件 {json_path}")
        sys.exit(1)

    data = json.loads(json_path.read_text(encoding='utf-8'))
    json_stem = json_path.stem

    if question_id is not None:
        item = next((x for x in data if x['id'] == question_id), None)
        if item is None:
            print(f"错误：未找到 id={question_id} 的题目")
            sys.exit(1)
        process_geometry_task(item, generic_knowledge, json_stem=json_stem)
    else:
        for item in data:
            process_geometry_task(item, generic_knowledge, json_stem=json_stem)
