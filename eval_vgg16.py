import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from net.inception_v2 import _reduced_kernel_size_for_small_input, inception_v2, inception_v2_arg_scope, inception_v2_base
from net.vgg import  vgg_16, vgg_arg_scope
import time
import os
from read_tfrecord import get_split, load_batch
from STNet import *
from TRL import *


slim = tf.contrib.slim

#State your log directory where you can retrieve your model
log_dir = '/log'

#Create a new evaluation log directory to visualize the validation process
log_eval = '/log_eval_test'


#State the dataset directory where the validation set is found
dataset_dir = '/NWPU_RESISC45/20p/'


#State the batch_size to evaluate each time, which can be a lot more than the training batch
batch_size = 64

#State the number of epochs to evaluate
num_epochs = 1

#Get the latest checkpoint file
checkpoint_file = tf.train.latest_checkpoint(log_dir)


#Create the file pattern of your TFRecord files so that it could be recognized later on
file_pattern = '_%s_*.tfrecord'

#State the image size you're resizing your images to. We will use the default inception size of 299.num_samples
image_size = 224
dropout_keep_prob = 1
stddev = 0.01

def run():
    #Create log_dir for evaluation information
    
    if not os.path.exists(log_eval):
        os.mkdir(log_eval)
    predict_file = open(log_eval+'/predictions.txt', 'w+')
    label_file = open(log_eval+'/labels.txt', 'w+')
    #Just construct the graph from scratch again
    with tf.Graph().as_default() as graph:
        net_init1 = tf.constant([0.8, -0.00, +0.0, +0.00, 0.8, +0.0])
        net_init2 = tf.constant([0.8, -0.00, +0.0, +0.00, 0.8, -0.0])
        tf.logging.set_verbosity(tf.logging.INFO)
        #Get the dataset first and load one batch of validation images and labels tensors. Set is_training as False so as to use the evaluation preprocessing
        with tf.variable_scope('Data_input'): 
            dataset = get_split('validation', dataset_dir, file_pattern, file_pattern_for_counting='')
            Aug_images, images, _, labels = load_batch(dataset, batch_size = batch_size, height = image_size, width = image_size, is_training = False, model_name = 'vgg')

        #Create some information about the training steps
        num_batches_per_epoch = dataset.num_samples / batch_size
        num_steps_per_epoch = num_batches_per_epoch

        # get feature for localization 
        with tf.variable_scope('Spatial_Transform_Nets'):  
            with slim.arg_scope(STNet_arg_scope(weight_decay=0.00005)):
                pre_images_1 = STNet(images, images, image_size, scope = 'Spatial_Transform_Nets_1', net_init = net_init1, stddev = 0.001)

                pre_images_2 = STNet(pre_images_1, pre_images_1, image_size, scope = 'Spatial_Transform_Nets_2', net_init = net_init2, stddev = 0.001)
        tf.summary.image('input_img',images,4)
        tf.summary.image('stn_img_1',pre_images_1,4)
        tf.summary.image('stn_img_2',pre_images_2,4)


        with slim.arg_scope(vgg_arg_scope(weight_decay=0.00005)):
            net_0, _ = vgg_16(images, num_classes = dataset.num_classes, scope = 'vgg_16_layer0', reuse = None, final_endpoint = 'conv5')

            net_1, _ = vgg_16(pre_images_1, num_classes = dataset.num_classes, scope = 'vgg_16_layer1', reuse = None, final_endpoint = 'conv5')

            net_2, _ = vgg_16(pre_images_2, num_classes = dataset.num_classes, scope = 'vgg_16_layer2', reuse = None, final_endpoint = 'conv5')


        with tf.variable_scope('Final_layer'):
            phi_I_0 = tf.einsum('ijkm,ijkn->imn',net_0,net_0)        
            phi_I_0 = tf.reshape(phi_I_0,[-1,512*512])
            phi_I_0 = tf.divide(phi_I_0,784.0)  
            y_ssqrt_0 = tf.multiply(tf.sign(phi_I_0),tf.sqrt(tf.abs(phi_I_0)+1e-12))
            z_l2_0 = tf.nn.l2_normalize(y_ssqrt_0, dim=1)
            logits_0 = slim.fully_connected(z_l2_0, dataset.num_classes, scope = 'full0', activation_fn = None)
            probabilities_0 = slim.softmax(logits_0, scope='Predictions')

            phi_I_1 = tf.einsum('ijkm,ijkn->imn',net_1,net_1)        
            phi_I_1 = tf.reshape(phi_I_1,[-1,512*512])
            phi_I_1 = tf.divide(phi_I_1,784.0)  
            y_ssqrt_1 = tf.multiply(tf.sign(phi_I_1),tf.sqrt(tf.abs(phi_I_1)+1e-12))
            z_l2_1 = tf.nn.l2_normalize(y_ssqrt_1, dim=1)
            logits_1 = slim.fully_connected(z_l2_1, dataset.num_classes, scope = 'full1', activation_fn = None)
            probabilities_1 = slim.softmax(logits_1, scope='Predictions')

            phi_I_2 = tf.einsum('ijkm,ijkn->imn',net_2,net_2)        
            phi_I_2 = tf.reshape(phi_I_2,[-1,512*512])
            phi_I_2 = tf.divide(phi_I_2,784.0)  
            y_ssqrt_2 = tf.multiply(tf.sign(phi_I_2),tf.sqrt(tf.abs(phi_I_2)+1e-12))
            z_l2_2 = tf.nn.l2_normalize(y_ssqrt_2, dim=1)
            logits_2 = slim.fully_connected(z_l2_2, dataset.num_classes, scope = 'full2', activation_fn = None)
            probabilities_2 = slim.softmax(logits_2, scope='Predictions')

        with tf.variable_scope('Final_layer_weights'):
            
            '''
            logits = tf.stack([logits_0, logits_1, logits_2], axis = 1)
            logits = tf.transpose(logits, [0, 2, 1])
            w = tf.get_variable("w_final", shape = [3, 1],initializer = tf.constant_initializer([2, 1, 1]))
            b = tf.get_variable("b_final", shape = [dataset.num_classes],initializer = tf.constant_initializer(0.0))
            logits = tf.einsum('ijk,kn->ijn',logits, w)
            logits = tf.squeeze(logits, [2])
            logits = tf.nn.bias_add(logits, b)
            '''
            #logits = 1*logits_0+0.8*logits_1+1*logits_2
            logits = 1.*logits_0+0.8*logits_1+0.5*logits_2









        # #get all the variables to restore from theaccuracy checkpoint file and create the saver function to restore
        variables_to_restore = slim.get_variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        def restore_fn(sess):
            return saver.restore(sess, checkpoint_file)

        #Just define the metrics to track without the loss or whatsoever
        with tf.variable_scope('Accruacy_Compute'):
            probabilities = slim.softmax(logits, scope='Predictions')
           # probabilities = 1*probabilities_0+5*probabilities_1+2*probabilities_2
            predictions = tf.argmax(probabilities, 1)
            accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, labels)
            metrics_op = tf.group(accuracy_update, probabilities)

        #Create the global step and an increment op for monitoring
        global_step = get_or_create_global_step()
        global_step_op = tf.assign(global_step, global_step + 1) #no apply_gradient method so manually increasing the global_step
        

        #Create a evaluation step function
        def eval_step(sess, metrics_op, global_step):
            '''
            Simply takes in a session, runs the metrics op and some logging information.
            '''
            start_time = time.time()
            _, global_step_count, accuracy_value, predictions_value, labels_value = sess.run([metrics_op, global_step_op, accuracy, predictions, labels])
            time_elapsed = time.time() - start_time

            #Log some information
            logging.info('Global Step %s: Streaming Accuracy: %.4f (%.2f sec/step)', global_step_count, accuracy_value, time_elapsed)
            print>>predict_file, 'predictions: \n', predictions_value
            print>>label_file, 'labels: \n', labels_value


            return accuracy_value

        #Define some scalar quantities to monitor
        tf.summary.scalar('Validation_Accuracy', accuracy)
        my_summary_op = tf.summary.merge_all()

        #Get your supervisor
        sv = tf.train.Supervisor(logdir = log_eval, summary_op = None, saver = None, init_fn = restore_fn)

        #Now we are ready to run in one session
        with sv.managed_session() as sess:
            for step in xrange(num_steps_per_epoch * num_epochs):
                sess.run(sv.global_step)
                #print vital information every start of the epoch as always
                if step % num_batches_per_epoch == 0:
                    logging.info('Epoch: %s/%s', step / num_batches_per_epoch + 1, num_epochs)
                    logging.info('Current Streaming Accuracy: %.4f', sess.run(accuracy))
                    
                #Compute summaries every 10 steps and continue evaluating
                if step % 10 == 0:
                    eval_step(sess, metrics_op = metrics_op, global_step = sv.global_step)
                    summaries = sess.run(my_summary_op)
                    sv.summary_computed(sess, summaries)
                    

                #Otherwise just run as per normal
                else:
                    eval_step(sess, metrics_op = metrics_op, global_step = sv.global_step)

            #At the end of all the evaluation, show the final accuracy
            logging.info('Final Streaming Accuracy: %.4f', sess.run(accuracy))


            logging.info('Model evaluation has completed! Visit TensorBoard for more information regarding your evaluation.')
            predict_file.close
            label_file.close

if __name__ == '__main__':
    run()
