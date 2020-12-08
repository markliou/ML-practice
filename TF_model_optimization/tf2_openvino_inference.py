import numpy as np 
from openvino import inference_engine as ie
from openvino.inference_engine import IENetwork, IECore

model_xml = 'saved_model.xml'
model_bin = 'saved_model.bin'

# loading the network
net = IENetwork(model=model_xml, weights=model_bin)

# define which device will be used 
plug_in = IECore()
exec_net = plug_in.load_network(net, "CPU") # load the network

# inferecing 
# dummy = np.random.random([1, 28, 28, 1])
dummy = np.random.random([1, 1, 28, 28]) #openvino use [n,c,h,w]
print(exec_net.input_info) #get the name of the input layer is "input_1"
exec_net.infer({'input_1':dummy})