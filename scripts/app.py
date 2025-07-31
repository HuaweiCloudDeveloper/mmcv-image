import os
import streamlit as st
import tempfile
from pathlib import Path
import subprocess
import time
from PIL import Image

# 固定模型路径（服务器内部路径）
SERVER_MODEL_CONFIG = "/home/mmdetection/configs/centernet/centernet_r18_8xb16-crop512-140e_coco.py"
SERVER_MODEL_WEIGHTS = "/home/mmdetection/centernet_resnet18_140e_coco_20210705_093630-bb5b3bf7.pth"

# 输出目录
OUTPUT_DIR = Path("/home/mmdetection/outputs")
VIS_DIR = OUTPUT_DIR / "vis"
os.makedirs(VIS_DIR, exist_ok=True)

# Streamlit 界面初始化
st.set_page_config(page_title="Detection WebApp", layout="wide")
st.title("MMDetection Web App")

# 侧边栏配置
with st.sidebar:
    st.header("配置参数")
    uploaded_file = st.file_uploader("上传图片", type=["jpg", "png", "jpeg"])
    score_thr = st.slider("置信度阈值", 0.0, 1.0, 0.3)
    device = st.selectbox("运行设备", ["cpu", "cuda:0"])

if uploaded_file:
    # 创建临时输入文件
    file_ext = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as f:
       f.write(uploaded_file.read())
       input_path = f.name

    # 生成唯一输出前缀
    timestamp = int(time.time())
    output_prefix = f"{Path(input_path).stem}_{timestamp}"

    # 根据文件类型构建执行命令
    if uploaded_file.type.startswith("image"):
       command = [
          "python", "/home/mmdetection/demo/image_demo.py",
          input_path,
          SERVER_MODEL_CONFIG,
          "--weights", SERVER_MODEL_WEIGHTS,
          "--device", device,
          "--pred-score-thr", str(score_thr),
          "--out-dir", str(OUTPUT_DIR)
       ]
       expected_output_dir = VIS_DIR


    # 执行检测命令
    with st.spinner("检测中，请稍候..."):
       result = subprocess.run(command, capture_output=True, text=True)

    # 删除临时文件
    Path(input_path).unlink(missing_ok=True)

    if result.returncode != 0:
       st.error(f"执行失败！错误信息：\n{result.stderr}")
    else:
       st.success("检测完成！正在加载结果...")

       # 图片处理
       if uploaded_file.type.startswith("image"):
          # 查找最近的图像文件
          output_files = sorted(expected_output_dir.glob("*"), key=os.path.getmtime, reverse=True)
          latest_file = next(
             (f for f in output_files if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']), None)

          if latest_file:
             # 显示检测结果
             img = Image.open(latest_file)
             st.image(img, caption="检测结果", use_column_width=True)

             # 提供下载按钮
             with open(latest_file, "rb") as fp:
                st.download_button(
                   label="📥 下载检测结果",
                   data=fp,
                   file_name=latest_file.name,
                   mime="image/jpeg" if latest_file.suffix.lower() in ['.jpg', '.jpeg'] else "image/png"
                )
          else:
             st.warning("⚠️ 未找到检测结果图像，请检查输出路径。")


else:
    st.info("请上传一张图片以开始检测。")
