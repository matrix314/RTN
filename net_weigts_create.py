import tensorflow as tf
from net.inception_v2 import inception_v2_arg_scope, inception_v2_base
from read_tfrecord import *
from net.vgg import  vgg_16, vgg_arg_scope


restore_file = '/home/chenzan/work/Version_3/net/vgg_16.ckpt'
save_file = '/home/chenzan/work/Version_3/10p/vgg_16_3layers.ckpt'
dataset_dir = '/home/chenzan/work/Data/NWPU_RESISC45/10p/'
image_size = 224
batch_size = 32
file_pattern = '_%s_*.tfrecord'
num_classes = 45

def run():

    dataset = get_split('train', dataset_dir, file_pattern=file_pattern, file_pattern_for_counting='')
    Aug_images, images, _, labels = load_batch(dataset, batch_size=batch_size, height = image_size, width = image_size, is_training = True, model_name='vgg')

 
    with slim.arg_scope(vgg_arg_scope()):  
        net, _ = vgg_16(images, scope = 'vgg_16', reuse = None, final_endpoint = 'conv5')


    with slim.arg_scope(vgg_arg_scope()):
        net_0, _ = vgg_16(images, scope = 'vgg_16_layer0', reuse = None, final_endpoint = 'conv5')    

    with slim.arg_scope(vgg_arg_scope()):
        net_1, _ = vgg_16(images, scope = 'vgg_16_layer1', reuse = None, final_endpoint = 'conv5')

    with slim.arg_scope(vgg_arg_scope()):
        net_2, _ = vgg_16(images, scope = 'vgg_16_layer2', reuse = None, final_endpoint = 'conv5')        


    variables = slim.get_variables_to_restore()
    l = (len(variables))/4
    variables_to_restore = variables[:l]
    variables_to_save2 = variables[l:2*l]
    variables_to_save3 = variables[2*l:3*l]
    variables_to_save4 = variables[3*l:4*l]
    
    #variables_to_save4 = variables[3*l:]
   

    restore = tf.train.Saver(variables_to_restore)  
    saver = tf.train.Saver(variables_to_save2 + variables_to_save3+variables_to_save4)  

    with tf.Session() as sess:
        restore.restore(sess, restore_file)
        for i in range(len(variables_to_restore)):
            assign_op = tf.assign(variables_to_save2[i], variables_to_restore[i])
            sess.run(assign_op)
            assign_op = tf.assign(variables_to_save3[i], variables_to_restore[i])
            sess.run(assign_op)
            assign_op = tf.assign(variables_to_save4[i], variables_to_restore[i])
            sess.run(assign_op)
            #assign_op = tf.assign(variables_to_save4[i], variables_to_restore[i])
            #sess.run(assign_op)
  
        saver.save(sess, save_file)


if __name__ == '__main__':
    run()