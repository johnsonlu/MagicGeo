from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
GEOMETRY_KNOWLEDGE = PROJECT_DIR / "prompt" / "geometry_knowledge.dat"


def test_geometry_knowledge_contains_quad_shape_section():
    content = GEOMETRY_KNOWLEDGE.read_text(encoding="utf-8")
    assert "四边形子类型约束清单" in content
    assert "平行四边形 ABCD（必写全部）" in content
    assert "矩形 ABCD（必写全部）" in content
    assert "正方形 ABCD（必写全部）" in content
    assert "菱形 ABCD（必写全部）" in content
    assert "parallel(A, B, D, C)" in content
    assert "ortho(A, B, B, C)" in content
    assert "equal_line(A, B, B, C)" in content


def test_geometry_knowledge_quad_section_is_after_coordinate_rules():
    content = GEOMETRY_KNOWLEDGE.read_text(encoding="utf-8")
    coord_section = content.index("# 几何坐标初始化原则")
    quad_section = content.index("# 四边形子类型约束清单")
    line_ratio = content.index("line_ratio(D, C, B, D, m):")
    assert line_ratio < coord_section < quad_section


def test_geometry_knowledge_contains_quad_coordinate_templates():
    content = GEOMETRY_KNOWLEDGE.read_text(encoding="utf-8")
    assert "坐标模板" in content
    assert "A=(0,0), B=(0,s), C=(s,s), D=(s,0)" in content
    assert "E=(e, s)" in content
