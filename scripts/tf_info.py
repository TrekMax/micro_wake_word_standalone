# pip install tensorflow flatbuffers netron
import tensorflow as tf

# 加载TFLite模型
interpreter = tf.lite.Interpreter(model_path="XLXL_V4.tflite")
interpreter.allocate_tensors()

# 获取输入详细信息
print("=== 输入信息 ===")
for i, input_detail in enumerate(interpreter.get_input_details()):
    print(f"输入 {i}:")
    print(f"  名称: {input_detail['name']}")
    print(f"  形状: {input_detail['shape']}")
    print(f"  数据类型: {input_detail['dtype']}")
    print(f"  量化信息: {input_detail['quantization']}")

# 获取输出详细信息
print("\n=== 输出信息 ===")
for i, output_detail in enumerate(interpreter.get_output_details()):
    print(f"输出 {i}:")
    print(f"  名称: {output_detail['name']}")
    print(f"  形状: {output_detail['shape']}")
    print(f"  数据类型: {output_detail['dtype']}")
    print(f"  量化信息: {output_detail['quantization']}")

# 获取操作符信息
print("\n=== 操作符信息 ===")
for op in interpreter._get_ops_details():
    print(f"操作符: {op['index']}, 类型: {op['op_name']}")
    