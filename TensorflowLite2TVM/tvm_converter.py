import numpy as np
import tvm.relay.frontend.tensorflow_parser as tr
import tvm.relay as relay
import tvm

# define the input and output tensor shape
input_tensor_shape = (1, 784)
output_tensor_shape = (1, 10)

# using the tvm relay frontend
model = 'cnn.pb'
graph_def = tr.TFParser(model).parse()
sym, params = relay.frontend.from_tensorflow(graph_def)

# compile
with relay.build_config(opt_level=3):
    graph, lib, params = relay.build(sym, target='llvm', params=params)

# save the comiled model
lib.export_library('cnn_tvm_lib.tar')
with open('cnn_tvm.json', 'w') as f:
    f.write(graph)
with open('cnn_tvm.params', 'wb') as f:
    f.write(relay.save_param_dict(params))

# loading test
load_lib = tvm.runtime.load_module('cnn_tvm_lib.tar')

