# thai_support.py
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import platform

def setup_thai_font():
    """
    ตั้งค่า matplotlib ให้รองรับภาษาไทย
    โดยไม่ต้องแก้โค้ดหลัก
    """

    font_paths = []

    # 🔹 Windows
    if platform.system() == "Windows":
        font_paths += [
            "C:/Windows/Fonts/THSarabunNew.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]

    # 🔹 Mac
    elif platform.system() == "Darwin":
        font_paths += [
            "/System/Library/Fonts/Supplemental/Thonburi.ttf",
        ]

    # 🔹 Linux / Streamlit Cloud
    else:
        font_paths += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]

    for path in font_paths:
        if os.path.exists(path):
            font_prop = fm.FontProperties(fname=path)
            plt.rcParams["font.family"] = font_prop.get_name()
            break

    # ป้องกันเครื่องหมายลบเพี้ยน
    plt.rcParams["axes.unicode_minus"] = False


def setup_pandas_display():
    """
    ทำให้ pandas แสดงผลภาษาไทยสวยขึ้น
    """
    import pandas as pd
    pd.set_option('display.unicode.east_asian_width', True)