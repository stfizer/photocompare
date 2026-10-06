# -*- coding: utf-8 -*-
"""生成应用图标 app.ico：左右两半色块 + 中间分割线（对比器意象）"""
import os
from PIL import Image, ImageDraw

S = 256
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)

# 圆角底板
d.rounded_rectangle([8, 8, S - 8, S - 8], radius=44, fill=(23, 23, 30, 255),
                    outline=(90, 120, 214, 255), width=6)

# 左半：冷色方块（原图）
d.rounded_rectangle([34, 64, 118, 192], radius=14, fill=(70, 130, 200, 255))
# 右半：暖色方块（修改后）
d.rounded_rectangle([138, 64, 222, 192], radius=14, fill=(230, 140, 80, 255))

# 中间分割线 + 手柄
d.line([128, 44, 128, 212], fill=(255, 255, 255, 255), width=8)
d.ellipse([112, 112, 144, 144], fill=(255, 255, 255, 255))

# 高光小箭头示意对比
d.polygon([(64, 100), (88, 128), (64, 156)], fill=(255, 255, 255, 220))
d.polygon([(196, 100), (172, 128), (196, 156)], fill=(255, 255, 255, 220))

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.ico")
img.save(out, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
print("icon saved:", out)
