import tensorflow as tf
import numpy as np

def analyze_micro_wakeword_model(model_path):
    """专门分析唤醒词模型的信息"""
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        print("🎯 MicroWakeWord 模型分析")
        print("=" * 50)
        
        # 输入详情
        input_details = interpreter.get_input_details()
        print("📥 输入层:")
        for inp in input_details:
            print(f"  • 名称: {inp['name']}")
            print(f"  • 形状: {inp['shape']}")
            print(f"  • 类型: {tf.dtypes.as_dtype(inp['dtype']).name}")
            if inp['quantization'] != (0.0, 0):
                print(f"  • 量化: {inp['quantization']}")
            print()
        
        # 输出详情
        output_details = interpreter.get_output_details()
        print("📤 输出层:")
        for out in output_details:
            print(f"  • 名称: {out['name']}")
            print(f"  • 形状: {out['shape']}")
            print(f"  • 类型: {tf.dtypes.as_dtype(out['dtype']).name}")
            if out['quantization'] != (0.0, 0):
                print(f"  • 量化: {out['quantization']}")
            
            # 对于唤醒词模型，输出通常是概率值
            if len(out['shape']) == 1 and out['shape'][0] > 1:
                print(f"  • 可能是 {out['shape'][0]} 个类别的概率输出")
            print()
        
        print("✅ 模型加载成功！")
        
    except Exception as e:
        print(f"❌ 模型分析失败: {e}")

# 使用
analyze_micro_wakeword_model("XLXL_V4.tflite")