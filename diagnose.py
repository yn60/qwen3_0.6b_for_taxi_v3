# wait_for_model.py
import time
from backend.llm.client import _get_client, _ensure_pipeline_ready

def wait_for_model_complete():
    """等待模型完全就绪"""
    print("⏳ 等待模型完全就绪...")
    
    client = _get_client()
    max_wait = 300  # 5分钟超时
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        # 检查模型状态
        if (_ensure_pipeline_ready(client) and 
            client._pipeline is not None and 
            client._pipeline.model is not None):
            
            # 测试真实推理
            try:
                from backend.taxi.state_utils import describe_state_for_llm, get_prompt
                
                EXAMPLE_DECODED_STATE = {
                    "taxi_row": 2, "taxi_col": 1,
                    "passenger_location": 4, "destination_index": 3,
                }
                
                state_desc = describe_state_for_llm(EXAMPLE_DECODED_STATE)
                prompt = get_prompt(state_desc)
                
                result = client.generate(prompt)
                if result.get('action') is not None:
                    print("✅ 模型完全就绪！")
                    print(f"等待时间: {time.time() - start_time:.0f}秒")
                    return True
                    
            except Exception as e:
                print(f"模型测试中: {e}")
        
        print(f"模型初始化中... 已等待 {int(time.time() - start_time)}秒")
        time.sleep(10)  # 每10秒检查一次
    
    print("❌ 模型初始化超时")
    return False

# 等待模型
if wait_for_model_complete():
    print("🎉 现在可以运行speed_test了！")
else:
    print("💥 需要重启Colab或检查网络")


# check_download.py
import os
import requests

def check_download_progress():
    """检查模型下载进度"""
    print("📥 检查模型下载进度")
    print("=" * 30)
    
    # Hugging Face缓存路径
    cache_path = "/root/.cache/huggingface/hub"
    model_path = os.path.join(cache_path, "models--Qwen")
    
    if os.path.exists(model_path):
        print("✅ 模型缓存目录存在")
        
        # 查找模型文件
        safetensors_files = []
        for root, dirs, files in os.walk(model_path):
            for file in files:
                if file.endswith('.safetensors'):
                    safetensors_files.append(os.path.join(root, file))
                elif file.endswith('.bin'):
                    print(f"找到模型文件: {os.path.join(root, file)}")
        
        if safetensors_files:
            print(f"✅ 找到 {len(safetensors_files)} 个模型文件")
            for file in safetensors_files[:3]:  # 显示前3个
                size = os.path.getsize(file) / (1024*1024*1024)  # GB
                print(f"  {os.path.basename(file)}: {size:.2f} GB")
        else:
            print("❌ 未找到模型权重文件")
    else:
        print("❌ 模型缓存目录不存在")

check_download_progress()

# quick_status.py
from backend.llm.client import _get_client

def quick_status():
    """快速状态检查"""
    print("🔍 快速状态检查")
    print("=" * 30)
    
    client = _get_client()
    
    print(f"Pipeline存在: {client._pipeline is not None}")
    print(f"模型存在: {getattr(client._pipeline, 'model', None) is not None if client._pipeline else False}")
    
    if client._pipeline and client._pipeline.model:
        try:
            # 尝试获取参数数量（如果模型已加载）
            param_count = sum(p.numel() for p in client._pipeline.model.parameters())
            print(f"模型参数: {param_count:,}")
            print("✅ 模型已加载完成")
        except:
            print("⚠️ 模型对象存在但可能未完全初始化")

quick_status()