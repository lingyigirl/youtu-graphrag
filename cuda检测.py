#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/11/3 13:41
# @Author : Administrator
# @File : cuda检测.py
# @Project : youtu-graphrag-main
# @Software: PyCharm

# 添加这个测试函数到您的代码中
def check_cuda_compatibility():
    """检查CUDA和PyTorch的兼容性"""
    print("\n检查CUDA兼容性...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA可用")
            print(f"   CUDA版本: {torch.version.cuda}")
            print(f"   显卡数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   显卡 {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("❌ CUDA不可用")
    except Exception as e:
        print(f"❌ CUDA检查失败: {e}")

# 在main函数中调用
if __name__ == "__main__":
    # ... 原有代码 ...
    # check_cuda_compatibility()
    import socket

    hostname = socket.gethostname()
    service_ip = socket.gethostbyname(hostname)
    print(f"service_ip={service_ip}")