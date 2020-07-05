#!/usr/bin/env python
# -*- coding: utf-8 -*-
  
import os
import tensorflow as tf
import multiprocessing
#  TF_MODELS_PATH = '/apdcephfs/private_forestlma/yanlongdong/Projects/'
TF_MODELS_PATH = '/apdcephfs/private_forestlma/yanlongdong/Projects/'
def initializer():
  model_name = 'faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28'
  #  path_to_ckpt = (TF_MODELS_PATH + model_name
                  #  + '/frozen_inference_graph.pb')
  path_to_ckpt = ( '../../tf_models/research'
                  + '/frozen_inference_graph.pb')

  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
      od_graph_def.ParseFromString(fid.read())
      tf.import_graph_def(od_graph_def, name='')

  global sess, tensor_dict, image_tensor
  with detection_graph.as_default():
    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(
      allow_growth=True)))
    ops = tf.get_default_graph().get_operations()
    #  for node in tf.get_default_graph().as_graph_def().node:
        #  print(node.name)
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    for x in all_tensor_names:
        print(x)
    tensor_dict = {}
    for key in [
      'num_detections', 'detection_boxes', 'detection_scores',
      'detection_classes', 'detection_masks'
    ]:
      tensor_name = key + ':0'
      if tensor_name in all_tensor_names:
        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
          tensor_name)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
def test():
    saver = tf.train.Saver()

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        saver.restore(sess, 
        '../../faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28/modle')
        sess.run(tf.local_variables_initializer())
        for n in input_graph_def.node:
            print(n.name)
if __name__ == '__main__':

    import warnings
    warnings.filterwarnings('ignore')
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    initializer()
    #  test()
    #  data = tf.random_normal()
    #  config = tf.estimator.RunConfig(model_dir='..')
    #  print(config)
