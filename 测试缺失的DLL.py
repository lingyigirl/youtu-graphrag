#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/11/3 09:48
# @Author : Administrator
# @File : 测试缺失的DLL.py
# @Project : youtu-graphrag-main
# @Software: PyCharm

# test_dll.py
import os
import sys
import ctypes


def check_vc_dlls():
    """检查常见的VC++运行时DLL"""
    dlls_to_check = [
        'vcruntime140.dll',  # VC++ 2015-2022
        'vcruntime140_1.dll',  # VC++ 2015-2022
        'msvcp140.dll',  # VC++ 2015-2022
        'vcruntime140.dll',  # VC++ 2013
        'msvcp120.dll',  # VC++ 2013
    ]

    print("检查VC++运行时DLL...")
    system32 = os.environ.get('SystemRoot', 'C:\\Windows') + '\\System32'
    syswow64 = os.environ.get('SystemRoot', 'C:\\Windows') + '\\SysWOW64'

    missing_dlls = []
    for dll in dlls_to_check:
        found = False
        # 检查System32 (64位系统)
        if os.path.exists(os.path.join(system32, dll)):
            print(f"✅ 找到 {dll} (System32)")
            found = True
        # 检查SysWOW64 (32位在64位系统)
        if os.path.exists(os.path.join(syswow64, dll)):
            print(f"✅ 找到 {dll} (SysWOW64)")
            found = True

        if not found:
            print(f"❌ 缺失 {dll}")
            missing_dlls.append(dll)

    return missing_dlls


def test_pytorch_import():
    """测试PyTorch相关导入"""
    print("\n测试PyTorch相关导入...")
    try:
        import torch
        print("✅ PyTorch导入成功")
        print(f"   PyTorch版本: {torch.__version__}")
        print(f"   CUDA可用: {torch.cuda.is_available()}")
    except Exception as e:
        print(f"❌ PyTorch导入失败: {e}")
        return False
    return True


def test_specific_imports():
    """测试具体的导入问题"""
    print("\n测试具体问题模块...")
    try:
        # 尝试导入可能出问题的模块
        import torch._C
        print("✅ torch._C 导入成功")
    except ImportError as e:
        print(f"❌ torch._C 导入失败: {e}")
        return False
    except Exception as e:
        print(f"⚠️  torch._C 其他错误: {e}")
        return False
    return True


if __name__ == "__main__":
    print("=== VC++运行时库诊断工具 ===\n")

    # 检查系统架构
    print(f"系统架构: {os.environ.get('PROCESSOR_ARCHITECTURE', 'Unknown')}")
    print(f"Python架构: {sys.version}")

    missing = check_vc_dlls()
    pytorch_ok = test_pytorch_import()
    specific_ok = test_specific_imports()

    print("\n=== 诊断结果 ===")
    if missing:
        print(f"❌ 缺失 {len(missing)} 个关键DLL文件")
        print("建议安装: Microsoft Visual C++ Redistributable 2015-2022")
    elif not pytorch_ok or not specific_ok:
        print("⚠️  DLL文件存在但可能版本不兼容")
        print("建议重新安装PyTorch或更新VC++运行库")
    else:
        print("✅ 所有检查通过")