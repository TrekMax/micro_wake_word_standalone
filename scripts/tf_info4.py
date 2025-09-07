import tensorflow as tf
import numpy as np

def analyze_neural_network_layers(model_path):
    """专门分析神经网络层的信息"""
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        tensor_details = interpreter.get_tensor_details()
        tensor_details.sort(key=lambda x: x['index'])
        
        print("🧠 神经网络层分析")
        print("=" * 70)
        
        # 分类不同类型的层
        conv_layers = []
        dense_layers = []
        other_layers = []
        
        for tensor in tensor_details:
            layer_name = tensor['name'].lower()
            
            if 'conv' in layer_name or 'conv2d' in layer_name:
                conv_layers.append(tensor)
            elif 'dense' in layer_name or 'fc' in layer_name or 'matmul' in layer_name:
                dense_layers.append(tensor)
            else:
                other_layers.append(tensor)
        
        # 打印卷积层信息
        if conv_layers:
            print("🎯 卷积层:")
            for tensor in conv_layers:
                shape = tensor['shape']
                if len(shape) == 4:  # [filter_height, filter_width, in_channels, out_channels]
                    print(f"  {tensor['name']}: {shape} "
                          f"(Kernel: {shape[0]}x{shape[1]}, "
                          f"In: {shape[2]}, Out: {shape[3]})")
        
        # 打印全连接层信息
        if dense_layers:
            print("\n🔗 全连接层:")
            for tensor in dense_layers:
                shape = tensor['shape']
                if len(shape) == 2:
                    print(f"  {tensor['name']}: {shape} (In: {shape[0]}, Out: {shape[1]})")
                elif len(shape) == 1:
                    print(f"  {tensor['name']}: {shape} (偏置项)")
        
        # 打印其他层信息
        if other_layers:
            print("\n🔧 其他层:")
            for tensor in other_layers:
                print(f"  {tensor['name']}: {tensor['shape']}")
        
        # 内存使用估算
        total_size = 0
        for tensor in tensor_details:
            if tensor['shape'].size > 0:
                param_count = np.prod(tensor['shape'])
                dtype_size = np.dtype(tf.dtypes.as_dtype(tensor['dtype']).as_numpy_dtype).itemsize
                total_size += param_count * dtype_size
        
        print(f"\n💾 估计内存使用: {total_size / 1024:.2f} KB")
        
    except Exception as e:
        print(f"❌ 错误: {e}")

# 使用
analyze_neural_network_layers("XLXL_V4.tflite")