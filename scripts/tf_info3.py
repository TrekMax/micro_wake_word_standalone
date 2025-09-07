import tensorflow as tf
import numpy as np

def print_model_layers_detailed(model_path):
    """打印模型的每一层详细信息"""
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        print("🔍 TFLite 模型层详细信息")
        print("=" * 60)
        
        # 获取所有张量详细信息
        tensor_details = interpreter.get_tensor_details()
        
        # 按索引排序
        tensor_details.sort(key=lambda x: x['index'])
        
        print(f"📊 总层数: {len(tensor_details)}")
        print("-" * 60)
        
        for i, tensor in enumerate(tensor_details):
            print(f"🏷️ 层 {i} (索引: {tensor['index']}):")
            print(f"  名称: {tensor['name']}")
            print(f"  形状: {tensor['shape']}")
            print(f"  数据类型: {tf.dtypes.as_dtype(tensor['dtype']).name}")
            
            # 量化信息
            if tensor['quantization'] != (0.0, 0):
                scale, zero_point = tensor['quantization']
                print(f"  量化参数: scale={scale}, zero_point={zero_point}")
            
            # 计算参数数量
            if tensor['shape'].size > 0:
                param_count = np.prod(tensor['shape'])
                print(f"  参数数量: {param_count:,}")
            
            # 检查是否是输入/输出层
            is_input = any(tensor['index'] == inp['index'] for inp in interpreter.get_input_details())
            is_output = any(tensor['index'] == out['index'] for out in interpreter.get_output_details())
            
            if is_input:
                print("  📥 输入层")
            if is_output:
                print("  📤 输出层")
            
            print("-" * 40)
            
    except Exception as e:
        print(f"❌ 错误: {e}")

# 使用
print_model_layers_detailed("XLXL_V4.tflite")