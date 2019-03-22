import numpy as np
import json
import glob

def get_look_vec_from_json(filename):
    with open(filename) as jf:
        j = json.load(jf)
    return eval(j['eye_details']['look_vec'])[:2]
pass 

maxi = [0,0]
mini = [0,0]
for i in glob.glob(r'imgs-lefteyes/*.json'):
    vec = get_look_vec_from_json(i)
    # print(vec)
    if vec[0] > maxi[0]:
        maxi[0] = vec[0]

    if vec[1] > maxi[1]:
        maxi[1] = vec[1]

    if vec[0] < mini[0]:
        mini[0] = vec[0]

    if vec[1] < mini[1]:
        mini[1] = vec[1]


pass 

print('max {}'.format(maxi))
print('min {}'.format(mini))

# final results 2019/3/21
# max [0.9367, 0.7634]
# min [-0.9388, -0.7626]

