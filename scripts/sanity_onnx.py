import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("exports/onnx/model.onnx", providers=["CPUExecutionProvider"])
x = np.zeros((3, 640, 640), dtype=np.float32)
outs = sess.run(None, {"image": x})
print([o.shape for o in outs])
print([o.dtype for o in outs])