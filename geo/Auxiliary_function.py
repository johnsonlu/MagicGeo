import os
import re
from dotenv import load_dotenv
from geo.Geometric_function import (
    dist,
    angle,
    angle_bisector,
    equal_line,
    ortho,
    online,
    midpoint,
    parallel,
    angle_relation,
    arc_midpoint,
    online_extension,
    online_inside,
    calc_var_from_dist,
    is_point_in_triangle,
    is_acute_triangle,
    is_point_out_triangle,
    line_ratio,
    check_coordinates_distinct,
    evaluate_radius_expression,
    is_r_dependent_expression,
    normalize_radius_symbol,
)

GEOMETRY_CONDITION_GLOBALS = {
    "dist": dist,
    "angle": angle,
    "angle_bisector": angle_bisector,
    "equal_line": equal_line,
    "ortho": ortho,
    "online": online,
    "midpoint": midpoint,
    "parallel": parallel,
    "angle_relation": angle_relation,
    "arc_midpoint": arc_midpoint,
    "online_extension": online_extension,
    "online_inside": online_inside,
    "calc_var_from_dist": calc_var_from_dist,
    "is_point_in_triangle": is_point_in_triangle,
    "is_acute_triangle": is_acute_triangle,
    "is_point_out_triangle": is_point_out_triangle,
    "line_ratio": line_ratio,
}


def eval_geometry_condition(execute_code, variables, coordinates):
    namespace = dict(GEOMETRY_CONDITION_GLOBALS)
    namespace["variables"] = variables
    namespace["coordinates"] = coordinates
    return eval(execute_code, namespace)
from openai import OpenAI

load_dotenv()

calculate_point_function = ['midpoint']
func_information = {
    'dist': 'dist(A,B,value)表示两点AB之间的距离为\{value\}',
    'angle': 'angle(A,B,C,value)表示角ABC的度数为value或者∠ABC=\{value\}°',
    'angle_bisector': '计算角的平分线',
    'equal_line': 'equal_line(A,B,C,D)表示AB等于CD',
    'ortho': '判断两个向量是否正交',
    'online': '判断点是否在直线上',
    'online_inside': '判断点是否在直线上',
    'online_extension': '判断点是否在直线的延长线上',
    'midpoint': '计算线段的中点',
    'parallel': '其中parallel(A,B,C,D)表示AB与CD平行',
    'angle_relation': '计算两个角之间的关系',
    'arc_midpoint': '计算圆弧的中点',
    "is_point_in_triangle": "判断点是否在三角形内",
    "is_acute_triangle": "判断三角形是否为锐角三角形",
    "is_point_out_triangle": "判断点是否在三角形外",
    "line_ratio": "判断线段的比例关系"
}

def LLM_check(text,condition):
    client = OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL", "https://api.deepseek.com"),
    )
    cond_key = condition[0]
    if cond_key == "dist" or cond_key == 'online_inside' or cond_key == 'ortho' or cond_key == 'equal_line' or cond_key =='online_extension' or cond_key =='angle' or cond_key =='midpoint' or cond_key =='angle_relation':
        return True
    cond_value = condition[1]
    cond_info = func_information[cond_key]
    params_str = "("
    for index in range(len(cond_value)):
        if index == len(cond_value) - 1:
            params_str += f"{cond_value[index]})"
        else:
            params_str += f"{cond_value[index]}, "
    query = f'题目：{text}，条件：{cond_key}{params_str},其中该条件的意思为{cond_info}，请据此意思判断从题目中是否能推出该条件，并只用"yes"或"no"回答。'
    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME", "deepseek-v4-flash"),
        messages=[
            {
                "role": "system",
                "content": ""
            },
            {"role": "user", "content": query},
        ],
        temperature=0.0,
        stream=False
    )

    # 提取响应中的 yes/no 信息
    condition_satisfied = response.choices[0].message.content.strip().lower()
    if "yes" in condition_satisfied:
        return True
    else:
        return False



def _split_point_names(group):
    names = []
    for segment in re.split(r"[,，、]", group):
        segment = segment.strip()
        if not segment:
            continue
        if len(segment) > 1 and segment.isalpha() and segment.isupper():
            names.extend(list(segment))
        else:
            names.append(segment)
    return names


_ON_CIRCLE_PATTERNS = (
    re.compile(r"点([A-Z](?:、[A-Z])*)在[⊙]?O上"),
    re.compile(r"([A-Z](?:[,，、][A-Z])*)是[⊙]?O上的(?:点|三点|不同的三点)"),
    re.compile(r"([A-Z])是[⊙]?O上一点"),
    re.compile(r"([A-Z])为圆上一点"),
    re.compile(r"点([A-Z])在(?:劣弧|优弧|弧)"),
    re.compile(r"([A-Z])在弧"),
    re.compile(r"三角形([A-Z]{3})内接于"),
    re.compile(r"四边形\s*([A-Z]{4})内接于"),
    re.compile(r"圆内接四边形([A-Z]{4})"),
    re.compile(r"正五边形([A-Z]{5})内接于"),
    re.compile(r"([A-Z]{2})是[⊙]?O的直径"),
    re.compile(r"([A-Z]{2})是[⊙]?O的半径"),
)
_OUTSIDE_CIRCLE_PATTERN = re.compile(r"点([A-Z])是[⊙]?O外")


def _extract_on_circle_points(text):
    if not re.search(r"[⊙]?O|圆", text):
        return set()

    outside_points = set(_OUTSIDE_CIRCLE_PATTERN.findall(text))
    on_circle = set()
    for pattern in _ON_CIRCLE_PATTERNS:
        for match in pattern.finditer(text):
            on_circle.update(_split_point_names(match.group(1)))

    return on_circle - outside_points


def _has_dist_on_circle(conditions, center, point):
    for function_name, params in conditions:
        if function_name != "dist" or len(params) < 3:
            continue
        if params[0] == center and params[1] == point:
            return True
    return False


def add_on_circle_dist_constraints(text, coordinates, conditions, radius=None):
    if "O" not in coordinates:
        return conditions

    dist_target = "r"
    if radius is not None:
        dist_target = str(radius)

    added = []
    for point in sorted(_extract_on_circle_points(text)):
        if point == "O" or point not in coordinates:
            continue
        if _has_dist_on_circle(conditions, "O", point):
            continue
        params = ["O", point, dist_target]
        if not check_function_format("dist", params, radius):
            continue
        conditions.append(["dist", params])
        added.append(point)

    if added:
        print(f"\n added on-circle dist constraints for: {', '.join(added)}\n")

    return conditions


def _parse_conditions_text(conditions_str, radius=None):
    conditions = []
    calculate_point_conditions = []
    lines = conditions_str.strip().split('\n')

    for line in lines:
        if not line.strip():
            continue
        parts = re.split(r",(?=\s*')", line.strip())
        for part in parts:
            part = part.strip().rstrip(',')
            match = re.match(r"'(\w+)':\s*(\w+)\((.*)\)", part)
            if match:
                condition_type = match.group(1)
                function_name = match.group(2)
                params = [param.strip() for param in re.split(r',\s*(?![^(]*\))', match.group(3))]
                if check_function_format(function_name, params, radius):
                    if condition_type in calculate_point_function:
                        calculate_point_conditions.append([function_name, params])
                    else:
                        conditions.append([function_name, params])
    return conditions, calculate_point_conditions


def _coord_is_numeric(value):
    return not isinstance(value, str)


def _segment_axis(dx, dy, eps=1e-9):
    if abs(dy) < eps:
        return "horizontal"
    if abs(dx) < eps:
        return "vertical"
    return "diagonal"


def fix_online_inside_coordinates(coordinates, _variables, conditions):
    """Align segment-point parameterization with the segment direction.

    Square problems often place E on the wrong edge, e.g. E=(0.5, e) when BC is
    horizontal and E should be (e, 0.5).
    """
    for cond in conditions:
        if cond[0] not in {"online_inside", "online", "online_extension"}:
            continue
        params = cond[1]
        if len(params) != 3:
            continue
        point_name, seg_a, seg_b = params
        if point_name not in coordinates:
            continue
        if seg_a not in coordinates or seg_b not in coordinates:
            continue

        seg_start = coordinates[seg_a]
        seg_end = coordinates[seg_b]
        point = coordinates[point_name]
        if not all(_coord_is_numeric(v) for v in seg_start + seg_end):
            continue

        ax, ay = float(seg_start[0]), float(seg_start[1])
        bx, by = float(seg_end[0]), float(seg_end[1])
        px, py = point[0], point[1]
        px_var = isinstance(px, str)
        py_var = isinstance(py, str)
        if px_var == py_var:
            continue

        var_name = px if px_var else py
        axis = _segment_axis(bx - ax, by - ay)
        if axis == "horizontal":
            if px_var:
                continue
            fixed_y = ay
            coordinates[point_name] = (var_name, fixed_y)
            print(
                f"[fix] {point_name} on {seg_a}{seg_b}: "
                f"({px}, {py}) -> ({var_name}, {fixed_y})"
            )
        elif axis == "vertical":
            if py_var:
                continue
            fixed_x = ax
            coordinates[point_name] = (fixed_x, var_name)
            print(
                f"[fix] {point_name} on {seg_a}{seg_b}: "
                f"({px}, {py}) -> ({fixed_x}, {var_name})"
            )


def convert_conditions(text, variables, coordinates, conditions_data, radius=None):
    conditions = []
    calculate_point_conditions = []

    if isinstance(conditions_data, dict):
        for cid, cond in conditions_data.items():
            if not cond or len(cond) < 2:
                continue
            function_name = cond[0]
            params = cond[1:]
            if check_function_format(function_name, params, radius):
                if cid in calculate_point_function:
                    calculate_point_conditions.append([function_name, params])
                else:
                    conditions.append([function_name, params])
    elif isinstance(conditions_data, str):
        conditions, calculate_point_conditions = _parse_conditions_text(conditions_data, radius)
    else:
        print(f"不支持的 conditions_data 类型: {type(conditions_data)}")
        return coordinates, variables, [], []

    fix_online_inside_coordinates(coordinates, variables, conditions)

    for cond in reversed(conditions):
        # check wirh LLM whether the condition is right or not
        condition_right = LLM_check(text,cond)
        if not condition_right:
            conditions.remove(cond)
            print("\n check with LLM, delete condition:\n")
            print(cond)

    conditions = add_on_circle_dist_constraints(
        text, coordinates, conditions, radius=radius
    )

                
# delete conditions that have already satisfied.
    conditions_excute = []
    for cond in reversed(conditions):
        cond_key = cond[0]
        cond_value = cond[1]
        if cond_key == 'equal_line' and len(cond_value) == 3:
            cond_key = 'dist'
        # check whether the condition is already satisifed or not
        params_str='variables,'
        if cond_key == 'dist' or cond_key == 'angle' or cond_key == 'angle_relation' or cond_key == 'line_ratio':
            for index in range(len(cond_value)):
                if index == len(cond_value) - 1:
                    # params_str += f"{int(cond_value[index])/10}"
                    params_str += f"{cond_value[index]},coordinates"
                else:
                    params_str += f"coordinates['{cond_value[index]}'],"
        else:
            for index in range(len(cond_value)):
                if index == len(cond_value) - 1:
                    # params_str += f"{int(cond_value[index])/10}"
                    params_str += f"coordinates['{cond_value[index]}'],coordinates"
                else:
                    params_str += f"coordinates['{cond_value[index]}'],"

        excute_code=f"{cond_key}({params_str})"

        try:
            # 尝试执行条件代码
            is_satisfied, is_calculate = eval_geometry_condition(
                excute_code, variables, coordinates
            )
            if is_calculate:
                if is_satisfied:
                    conditions.remove(cond)
                    print("\n already satiified, delete condition:\n")
                    print(excute_code)
            else:
                conditions_excute.append(excute_code)
        except Exception as e:
            print(excute_code)
            print("不能执行条件代码in convert_conditions")
            conditions=None
            break
    conditions_excute_reverse = []
    for excute_code in reversed(conditions_excute):
        conditions_excute_reverse.append(excute_code)

    #可计算点替换为函数    
    for cond in calculate_point_conditions:
        cond_key = cond[0]
        cond_value = cond[1]
        depend_point = cond_value[0]
        if coordinates[depend_point][0] in variables:
            del variables[coordinates[depend_point][0]]
        if coordinates[depend_point][1] in variables:
            del variables[coordinates[depend_point][1]]
        del coordinates[depend_point]
        dependences=f"calc_{cond_key}(variables,"
        for index in range(1, len(cond_value)):
            dependences+=f"coordinates['{cond_value[index]}'],"
        coordinates[depend_point] = [dependences+'coordinates)']
    print(coordinates)
    print(variables)

    print(calculate_point_conditions)
    print(conditions_excute_reverse)
    return coordinates,variables,calculate_point_conditions,conditions_excute_reverse



"""利用对应格式的字典生成变量"""
def convert_coordinates(coordinates, radius=None):
    def register_radius_variable():
        if "r" not in variable_names:
            variable_names["r"] = [0, False]

    def parse_value(value):
        value = normalize_radius_symbol(value)
        if value == "r":
            if radius is not None:
                return radius, False
            register_radius_variable()
            return "r", True
        if value == "-r":
            if radius is not None:
                return -radius, False
            register_radius_variable()
            return "-r", True
        if is_r_dependent_expression(value):
            if radius is not None:
                return evaluate_radius_expression(value, radius), False
            register_radius_variable()
            return value, True
        try:
            return float(value), False
        except ValueError:
            if value not in variable_names:
                variable_names[value] = [0, False]
            return value, True

    converted_coords = {}
    variable_names = {}

    for key, value in coordinates.items():
        if isinstance(value, (tuple, list)):
            coord_value = []
            for v in value:
                parsed, _is_string = parse_value(str(v).strip())
                coord_value.append(parsed)
            converted_coords[key] = tuple(coord_value)
        else:
            parts = value.split(',')
            coord_value = []
            for part in parts:
                parsed, _is_string = parse_value(part.strip())
                coord_value.append(parsed)
            converted_coords[key] = tuple(coord_value)

    return converted_coords, variable_names




def is_number(s, radius=None):
    try:
        # 尝试将输入转换为浮点数
        return float(s), True
    except ValueError:
        if s == 'r':
            # 如果 s 是 'r'，根据给定的半径返回 1.0 或者 None
            return radius if radius is not None else 1.0, True
        elif s == '-r':
            # 如果 s 是 '-r'，根据给定的半径返回 -1.0 或者 None
            return -radius if radius is not None else -1.0, True
        else:
            # 其他情况，返回原始字符串
            return s, False


def check_function_format(function_name,params, radius=None):
    """
        检查输入的函数名和参数格式是否正确。

        参数:
            function_name (str): 函数的名称，可能是 'angle' 或 'dist'。
            params (list): 与函数相关的参数列表。

        返回:
            bool: 如果参数格式正确且最后一个参数是一个数值，返回 True；否则返回 False。
        """
    if function_name=="angle":
        if len(params)==4:
            value,is_number_value = is_number(params[3])
            params[3]=value
            return is_number_value
        else:
            return False
    elif function_name=="dist":
        if len(params)==3:
            value,is_number_value = is_number(params[2], radius)
            params[2]=value
            return is_number_value
        else:
            return False
    elif function_name == "line_ratio":
        if len(params) == 5:
            value, is_number_value = is_number(params[4])  # 检查倍数因子 k 是否为数字
            params[4] = value
            return is_number_value
        else:
            return False
    elif function_name == "equal_line":
        # 检查 equal_line 是否有 4 个参数
        return len(params) == 4 or len(params) == 3
    # elif function_name == "equal_segment":
    #     # 检查 equal_segment 是否有 3 个参数
    #     return len(params) == 3
    elif function_name == "ortho":
        # 检查 ortho 是否有 4 个参数
        return len(params) == 4
    elif function_name == "online":
        # 检查 online 是否有 3 个参数
        return len(params) == 3
    elif function_name == "online_extension":
        # 检查 online 是否有 3 个参数
        return len(params) == 3
    elif function_name == "online_inside":
        # 检查 online 是否有 3 个参数
        return len(params) == 3
    elif function_name == "midpoint":
        # 检查 midpoint 是否有 3 个参数
        return len(params) == 3
    elif function_name == "angle_bisector":
        # 检查 angle_bisector 是否有 3 个参数
        return len(params) == 5
    elif function_name == "parallel":
        # 检查 parallel 是否有 3 个参数
        return len(params) == 4
    elif function_name == "is_point_in_triangle":
        # 检查 is_point_in_triangle 是否有 4 个参数
        return len(params) == 4
    elif function_name == "is_point_out_triangle":
        # 检查 is_point_in_triangle 是否有 4 个参数
        return len(params) == 4
    elif function_name == "is_acute_triangle":
        # 检查 is_acute_triangle 是否有 3 个参数
        return len(params) == 3
    elif function_name == "angle_relation":
        # 检查 angle_relation 是否有 7 个参数
        if len(params) == 7:
            value, is_number_value = is_number(params[6])  # 检查第七个参数
            params[6] = value
            return is_number_value
        else:
            return False
    elif function_name == "arc_midpoint":
        # 检查 arc_midpoint 是否有 3 个参数
        return len(params) == 3
    # else:
    #     print("格式错误，请停止重新启动")
    #     exit()
    else:
        print(f"{function_name} check_function_format 格式错误，请停止重新启动")
        return False


def check_condition_break(condition_code,coordinates,variables):
    """
    检查给定的条件代码是否有效。

    参数：
    condition_code -- 包含条件代码的字符串

    返回值：
    如果条件代码有效，则返回 True；否则返回 False
    """
    if not condition_code:
        return False
    for excute_code in condition_code:
        try:
            # 尝试执行条件代码
            is_satisfied, is_calculate = eval_geometry_condition(
                excute_code, variables, coordinates
            )
            if is_calculate:
                if not is_satisfied:
                    # print(excute_code)
                    return True
        except:
            # 如果执行条件代码时发生异常，则条件代码无效
            # print(excute_code)
            print(f"不能执行条件代码 {excute_code} in check_condition_break")
            return True
    if not check_coordinates_distinct(coordinates, variables):
        return True
    return False


def calc_deduct_var_values(deduct_var, variables):
    # print(variables)
    func_name = variables['depend'][deduct_var]['func_name']
    params = variables['depend'][deduct_var]['params']
    excute_code = f'{func_name}('
    for index in range(len(params)):
        if index == len(params) - 1:
            if isinstance(params[index], str):
                excute_code += f"variables['{params[index]}'][0]" + ')'
            else:
                excute_code += f"{params[index]}" + ')'
        else:
            if isinstance(params[index],str):
                excute_code += f"variables['free']['{params[index]}'][0],"
            else:
                excute_code += f"{params[index]},"

    try:
        result = eval(excute_code)
        return result
    except:
        print(excute_code)
        print("不能执行条件代码 in calc_deduct_var_values")
        return None
