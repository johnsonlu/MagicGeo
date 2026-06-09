import re
import tkinter as tk
from tkinter import ttk
from pdf2image import convert_from_path
from PIL import Image, ImageTk
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
from pathlib import Path

def get_latex_code(user_input):
    latex_code = user_input
    start_index = latex_code.find("\\documentclass")
    end_index = latex_code.find("\\end{document}") + len('\\end{document}')
    latex_code = latex_code[start_index:end_index]
    return latex_code


def for_render_code(latex_code):
    # Matplotlib's PGF backend injects its own \documentclass.
    latex_code = re.sub(r"\\documentclass(?:\[[^\]]*\])?\{[^}]+\}\s*", "", latex_code)

    if "\\usepackage{tikz}" not in latex_code:
        latex_code = "\\usepackage{tikz}\n" + latex_code

    if "\\usetikzlibrary{calc}" not in latex_code:
        latex_code = latex_code.replace(
            "\\usepackage{tikz}",
            "\\usepackage{tikz}\n\\usetikzlibrary{calc}",
        )

    if "\\begin{document}" in latex_code:
        preamble, body = latex_code.split("\\begin{document}", 1)
    else:
        draw_start = re.search(r"\\begin\{(?:tikzpicture|pgfpicture)\}", latex_code)
        if draw_start:
            preamble, body = latex_code[: draw_start.start()], latex_code[draw_start.start() :]
        else:
            preamble, body = latex_code, ""

    body = body.replace("\\end{document}", "").strip()
    preamble = preamble.strip()
    tikz_index = preamble.find("\\usepackage{tikz}")
    if tikz_index != -1:
        preamble = preamble[tikz_index:]

    if not body:
        return preamble

    return f"{preamble}\n\\begin{{document}}\n{body}\n\\end{{document}}"


def write_latex_debug(latex_code, output_path):
    debug_path = Path(output_path).with_suffix('.txt')
    debug_path.write_text(latex_code, encoding='utf-8')


def render_latex_to_pdf(latex_code, output_file):
    write_latex_debug(latex_code, output_file)
    plt.rc('text', usetex=False)
    plt.rc('pgf', rcfonts=False, preamble=latex_code)


    fig, ax = plt.subplots(figsize=(4, 4))  # Adjust the size as needed
    ax.axis('off')  # Hide axes
    fig.savefig(output_file)
    plt.close(fig)
    print(f"已存入文件{output_file}")


def show_pdf(file_path):
    # 将PDF转换为图像
    images = convert_from_path(file_path)

    root = tk.Tk()
    root.title("PDF Viewer")

    for image in images:
        # 获取图像的原始尺寸
        width, height = image.size

        # 创建一个与图像尺寸相匹配的画布
        canvas = tk.Canvas(root, width=width, height=height)
        canvas.pack()

        # 将图像调整为画布大小
        photo = ImageTk.PhotoImage(image)

        # 在画布上创建图像
        canvas.create_image(0, 0, anchor="nw", image=photo)
        canvas.image = photo  # 保持对图像的引用，防止被垃圾回收

        # 更新画布以显示图像
        canvas.update()
    root.mainloop()





