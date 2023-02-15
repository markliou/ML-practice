import numpy as np
import tensorflow as tf

def _checkFeatureMapSize(x, steps):
    # due to the latent space must be constructed with the same size,
    # the feature map after squeezed must be obeyed this rule
    target_shape = x.shape
    
    # assert (len(target_shape) == 3)
    assert (target_shape[0] % steps == 0) 
    assert (target_shape[1] % steps == 0)
pass 

def sqeezeSingle2DFeatureMap(x, steps = 2):
    _checkFeatureMapSize(x, steps)

    sqeezed_feature_map = [] 
    for start_x in range(steps):
        for start_y in range(steps):
            sqeezed_feature_map.append(x[start_y::steps, start_x::steps]) 
        pass
    pass 
    return sqeezed_feature_map
pass 

def sqeeze2DFeatureMap(x, steps = 2):
    _checkFeatureMapSize(x, steps)

    sqeezed_feature_map = [] 
    for start_x in range(steps):
        for start_y in range(steps):
            sqeezed_feature_map.append(x[start_y::steps, start_x::steps, ::]) 
        pass
    pass 
    return tf.concat(sqeezed_feature_map, axis=-1)
pass 

def unsqeezeSingle2DFeatureMap(xs, steps = 2):
    # direction obey the rule of row-to-column
    assert (len(xs) ** .5 * 10 % 10 == 0) # check if the feature map can be unsqeeze

    feature_shape = int(tf.Variable(xs[0][0].shape)) * steps

    def _makeIndeces(x, startx_p, starty_p):
        indices = []
        for xp in range(startx_p, feature_shape, steps):
            for yp in range(starty_p, feature_shape, steps):
                indices.append([xp, yp])
            pass 
        pass
        return indices
    pass

    unsqeeze_maps = tf.zeros([feature_shape, feature_shape])
    print(unsqeeze_maps)
    blocks = [ [i, j] for i in range(steps) for j in range(steps)]
    for x_ind in range(len(xs)):
        indices = _makeIndeces(xs[x_ind], blocks[x_ind][1], blocks[x_ind][0])
        unsqeeze_maps += tf.scatter_nd(indices, tf.reshape(xs[x_ind], [-1]), [feature_shape, feature_shape])
    pass
    return unsqeeze_maps
pass 

def unsqeeze2DFeatureMap(xs, steps = 2, channel = 3):
    sxs = tf.split(xs, [channel for c in range(int(xs.shape[-1])//channel)], -1)
    
    # direction obey the rule of row-to-column
    assert (len(sxs) ** .5 * 10 % 10 == 0) # check if the feature map can be unsqeeze
    
    feature_shape = int(tf.Variable(sxs[0].shape[0])) * steps

    def _makeIndeces(x, startx_p, starty_p):
        indices = []
        for xp in range(startx_p, feature_shape, steps):
            for yp in range(starty_p, feature_shape, steps):
                for cp in range(channel):
                    indices.append([xp, yp, cp])
                pass
            pass 
        pass
        return indices
    pass

    unsqeeze_maps = tf.zeros([feature_shape, feature_shape, channel])
    blocks = [ [i, j] for i in range(steps) for j in range(steps)]
    for x_ind in range(len(sxs)):
        indices = _makeIndeces(sxs[x_ind], blocks[x_ind][1], blocks[x_ind][0])
        unsqeeze_maps += tf.scatter_nd(indices, tf.reshape(sxs[x_ind], [-1]), [feature_shape, feature_shape, channel])
    pass
    return unsqeeze_maps
pass 

def main():
    size = 6
    
    # single channel
    a = tf.reshape(tf.Variable([float(i) for i in range(size ** 2)]), (size, size))
    print(a)
    b = sqeezeSingle2DFeatureMap(a)
    print(b)
    c = unsqeezeSingle2DFeatureMap(b)
    print(c)

    # multi-channel
    channel = 4
    a_rgb = tf.reshape(tf.Variable([float(i) for i in range(size ** 2 * channel)]), (size, size, channel))
    print(a_rgb)
    
    b_rgb = sqeeze2DFeatureMap(a_rgb)
    print(b_rgb)
    print(a_rgb.shape)
    print(b_rgb.shape)
    c_rgb = unsqeeze2DFeatureMap(b_rgb, channel = 4)
    print(c_rgb)

    # try to using tf.map
    batch = 5
    a_rgb_b = tf.reshape(tf.Variable([float(i) for i in range(size ** 2 * channel * batch)]), (batch, size, size, channel))
    print(a_rgb_b)
    def mapfn_a(x):
        return sqeeze2DFeatureMap(x, steps=2)
    pass
    b_rgb_b = tf.map_fn(fn=mapfn_a, elems=a_rgb_b)
    print(b_rgb_b)
    def mapfn_b(x):
        return unsqeeze2DFeatureMap(x, steps=2, channel=4)
    pass
    c_rgb_b = tf.map_fn(fn=mapfn_b, elems=b_rgb_b)
    print(c_rgb_b)

pass 

if __name__ == "__main__":
    main() 
pass 
