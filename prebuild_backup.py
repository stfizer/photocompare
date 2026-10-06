# -*- coding: utf-8 -*-
"""打包前备份：把当前 exe 挪为"上一版"，始终只保留一个旧版。"""
import glob
import os
import shutil

DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dist")
exes = glob.glob(os.path.join(DIST, "修图参数对比器.exe"))
if not exes:
    print("no previous build, skip backup")
    raise SystemExit(0)
cur = exes[0]
prev = os.path.join(DIST, "修图参数对比器_上一版.exe")
if os.path.exists(prev):
    os.remove(prev)  # 只保留一个上一版
shutil.move(cur, prev)
print(f"backed up previous build -> {os.path.basename(prev)}")
