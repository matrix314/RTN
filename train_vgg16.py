import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
from net.inception_v2 import inception_v2_arg_scope, inception_v2_base
from net.vgg import  vgg_16, vgg_arg_scope
import os
import time
from read_tfrecord import *
from STNet import *
slim = tf.contrib.slim
from TRL import *
from pairwise_loss import *

#================ DATASET INFORMATION ======================
#State dataset directory where the tfrecord files are located
dataset_dir = '/home/chenzan/work/Data/NWPU_RESISC45/20p/'

#State where your log file is at. If it doesn't exist, create it.
log_dir = '/home/chenzan/work/Version_3/NWPU_RESISC45/20p/log'

#State where your checkpoint file is

#checkpoint_file = '/home/chenzan/work/Version_3/UCMerced_LandUse/vgg_16_3layers.ckpt'
checkpoint_file = '/home/chenzan/work/Version_3/NWPU_RESISC45/20p/log_tmp_1/model.ckpt-5296'

#State the image size you're resizing your images to. We will use the default inception size of 299.
image_size = 224


#Create the file pattern of your TFRecord files so that it could be recognized later on
file_pattern = '_%s_*.tfrecord'

#================= TRAINING INFORMATION ==================
#State the number of epochs to train
num_epochs = 300

#State your batch size
batch_size = 32

#Learning rate information and configuration (Up to you to experiment)
num_epochs_before_decay = 10
dropout_keep_prob = 0.7
stddev = 0.01

#======================= TRAINING PROCESS =========================
def run():
    #Create the log directory here. Must be done here otherwise import will activate this unneededly.
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
   
    #Now we start to construct the graph and build our model
    with tf.Graph().as_default() as graph:
        net_init1 = tf.constant([0.8, -0.0, +0.0, +0.0, 0.8, +0.0])
        net_init2 = tf.constant([0.8, 0.0, +0.0, +0.0, 0.8, -0.0])
        tf.logging.set_verbosity(tf.logging.INFO) #Set the verbosity to INFO level

        #First create the dataset and load one batch
        with tf.variable_scope('Data_input'): 
            dataset = get_split('train', dataset_dir, file_pattern = file_pattern, file_pattern_for_counting='')
            Aug_images, images, _, labels = load_batch(dataset, batch_size=batch_size, height = image_size, width = image_size, is_training = False, model_name = 'vgg')                               
        #Know the number steps to take before decaying the learning rate and batches per epoch
        num_batches_per_epoch = int(dataset.num_samples / batch_size)
        num_steps_per_epoch = num_batches_per_epoch #Because one step is one batch processed
        decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)
        
        # get feature for localization 
        with tf.variable_scope('Spatial_Transform_Nets'):  
            with slim.arg_scope(STNet_arg_scope(weight_decay=0.00005)):
                pre_images_1 = STNet(images, images, image_size, scope = 'Spatial_Transform_Nets_1', net_init = net_init1, stddev = 0.001)

                pre_images_2 = STNet(pre_images_1, pre_images_1, image_size, scope = 'Spatial_Transform_Nets_2', net_init = net_init2, stddev = 0.001)
                tf.summary.image('input_img',images,1)
                tf.summary.image('stn_img_1',pre_images_1,1)
                tf.summary.image('stn_img_2',pre_images_2,1)


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
            
            logits = logits_0+0.8*logits_1+0.5*ogits_2



        #Define tdecay_stepshe scopes that you want to exclude for restoration

        exclude = ['Final_layer_weights','Spatial_Transform_Nets','Final_layer'] # 'Spatial_Transform_Nets','Final_layer' 'vgg_16/fc6/BatchNorm/moving_variance'
        '''
        for i in range(5):
            for j in range(3):
                exclude.append('vgg_16/conv{0}/conv{1}_{2}/BatchNorm'.format(i+1, i+1, j+1))
        '''
        exclude = None
        variables_to_restore = slim.get_variables_to_restore(exclude = exclude)

        with tf.variable_scope('Loss_Compute'):
            #Perform one-hot-enco1ding of the labels (Try one-hot-encoding within the load_batch function!)
            one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)

            #Performs the equivalent to tf.nn.sparse_softmax_cross_entropy_with_logits but enhanced with checks
            loss_intra = slim.losses.softmax_cross_entropy(onehot_labels = one_hot_labels, logits = logits)
            #total_loss = loss

            loss_intra_0 = slim.losses.softmax_cross_entropy(onehot_labels = one_hot_labels, logits = logits_0)
            loss_intra_1 = slim.losses.softmax_cross_entropy(onehot_labels = one_hot_labels, logits = logits_1)
            loss_intra_2 = slim.losses.softmax_cross_entropy(onehot_labels = one_hot_labels, logits = logits_2)
            
            loss_inter_1 = pairwise_loss(probabilities_0, probabilities_1, one_hot_labels, margin_1 = 0.05, weights = 0.1)
            loss_inter_2 = pairwise_loss(probabilities_1, probabilities_2, one_hot_labels, margin_1 = 0.05, weights = 0.1)
            slim.losses.add_loss(loss_inter_1)
            slim.losses.add_loss(loss_inter_2)
            
            
            
            
            
            

            total_loss = slim.losses.get_total_loss()    #obtain the regularization losses as well


            
            tf.summary.scalar('losses/loss_intra_0', loss_intra_0)
            tf.summary.scalar('losses/loss_intra_1', loss_intra_1)
            tf.summary.scalar('losses/loss_intra_2', loss_intra_2)
            tf.summary.scalar('losses/loss_intra', loss_intra)
            
            tf.summary.scalar('losses/loss_inter_1', loss_inter_1)
            tf.summary.scalar('losses/loss_inter_2', loss_inter_2)  
            
            
                    
            
            
            tf.summary.scalar('losses/Loss', total_loss)
       

        #Create the global step for monitoring the learning_rate and training.
        global_step = get_or_create_global_step()
        
        #Define your exponentially decaying learning rate
        lr_layer_1 = tf.train.exponential_decay(
            learning_rate = 0.0001,  #first 0.0, then 0.001, 0.0000000001
            global_step = global_step,
            decay_steps = decay_steps,
            decay_rate = 0.5,
            staircase = True)
        lr_layer_2 = tf.train.exponential_decay(
            learning_rate = 0.001,#first 0.00001, then 0.000001, 0.000001
            global_step = global_step,
            decay_steps = decay_steps,
            decay_rate = 0.5,
            staircase = True)
        lr_layer_3 = tf.train.exponential_decay(
            learning_rate = 0.001,  #first 0.0, then 0.001
            global_step = global_step,
            decay_steps = decay_steps,
            decay_rate = 0.5,
            staircase = True)


        #Now we can define the optimizer that takes on the learning rate and create the train_op

        #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #with tf.control_dependencies(update_ops):
            
        var_list1 = tf.trainable_variables()[:16]
        var_list2 = tf.trainable_variables()[16:-6]
        var_list3 = tf.trainable_variables()[-6:]
        opt1 = tf.train.GradientDescentOptimizer(lr_layer_1)
        opt2 = tf.train.MomentumOptimizer(lr_layer_2, momentum=0.9)
        opt3 = tf.train.MomentumOptimizer(lr_layer_3, momentum=0.9)

        grads = tf.gradients(total_loss, var_list1 + var_list2 + var_list3)
        grads1 = grads[:len(var_list1)]
        grads2 = grads[len(var_list1):-len(var_list3)]
        grads3 = grads[-len(var_list3):]
        train_op1 = opt1.apply_gradients(zip(grads1, var_list1), global_step = global_step)
        train_op2 = opt2.apply_gradients(zip(grads2, var_list2), global_step = global_step)
        train_op3 = opt3.apply_gradients(zip(grads3, var_list3), global_step = global_step)
        train_op = tf.group(train_op1, train_op2, train_op3)
            

        

        #State the metrics that you wa[nt to predict. We get a predictions that is not one_hot_encoded.


        with tf.variable_scope('Accruacy_Compute'):
            #probabilities = (probabilities_0+probabilities_1+probabilities_2)/3
            probabilities = slim.softmax(logits, scope='Predictions')
            predictions = tf.argmax(probabilities, 1)
            accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, labels)
            
            tf.summary.scalar('accuracy', accuracy)

            predictions_0 = tf.argmax(probabilities_0, 1)
            predictions_1 = tf.argmax(probabilities_1, 1)
            predictions_2 = tf.argmax(probabilities_2, 1)
            accuracy_0, accuracy_update_0 = tf.contrib.metrics.streaming_accuracy(predictions_0, labels)
            accuracy_1, accuracy_update_1 = tf.contrib.metrics.streaming_accuracy(predictions_1, labels)
            accuracy_2, accuracy_update_2 = tf.contrib.metrics.streaming_accuracy(predictions_2, labels)
            tf.summary.scalar('accuracy_0', accuracy_0)
            tf.summary.scalar('accuracy_1', accuracy_1)
            tf.summary.scalar('accuracy_2', accuracy_2)

            metrics_op = tf.group(accuracy_update, probabilities, accuracy_update_0, probabilities_0, accuracy_update_1, probabilities_1, accuracy_update_2, probabilities_2)


        #Now finally create all the summaries you need to monitor and group them into one summary op.
        
        
        
        tf.summary.scalar('learning_rate_1', lr_layer_3)
        my_summary_op = tf.summary.merge_all()

        #Now we need to create a training step function that runs both the train_op, metrics_op and updates the global_step concurrently.
        def train_step(sess, train_op, global_step, total_loss = total_loss):
            '''
            Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
            '''
            #Check the time for each sess run
            start_time = time.time()
            _, total_loss, global_step_count, _ = sess.run([train_op, total_loss, global_step, metrics_op])
            time_elapsed = time.time() - start_time

            #Run the logging to print some results
            logging.info('global step %s: loss: %.4f (%.2f sec/step)', global_step_count, total_loss, time_elapsed)

            return total_loss, global_step_count

        #Now we create a saver function that actually restores the variables from a checkpoint file in a sess
        saver = tf.train.Saver(variables_to_restore)
        def restore_fn(sess):
            return saver.restore(sess, checkpoint_file)

        #Define your supervisor for ruadd_image_summariesnning a managed session. Do not run the summary_op automatically or else it will consume too much memory
        sv = tf.train.Supervisor(logdir = log_dir, summary_op = None, init_fn = restore_fn, global_step = global_step)


        #Run the managed session
        with sv.managed_session() as sess:
            #sess.run(tf.global_variables_initializer())
            #restore_fn(sess)
            for step in xrange(num_steps_per_epoch * num_epochs):
                #At the start of every epoch, show the vital information:
                if step % num_batches_per_epoch == 0:
                    logging.info('Epoch %s/%s', step/num_batches_per_epoch + 1, num_epochs)
                    learning_rate_value, accuracy_value = sess.run([lr_layer_2, accuracy])
                    logging.info('Current Learning Rate: %s', learning_rate_value)
                    logging.info('Current Streaming Accuracy: %s', accuracy_value)

                    # optionally, print your logits and predictions for a sanity check that things are going fine.
                    logits_value, probabilities_value, predictions_value, labels_value = sess.run([logits, probabilities, predictions, labels])
                    print 'logits: \n', logits_value
                    print 'Probabilities: \n', probabilities_value
                    print 'predictions: \n', predictions_value
                    print 'Labels:\n:', labels_value

                #Log the summaries every 10 step.
                if step % 10 == 0:
                    total_loss, _ = train_step(sess, train_op, sv.global_step)
                    summaries = sess.run(my_summary_op)
                    sv.summary_computed(sess, summaries)
                    
                #If not, simply run the training step
                else:
                    total_loss, _ = train_step(sess, train_op, sv.global_step)

            #We log the final training loss and accuracy
            logging.info('Final Loss: %s', total_loss)
            logging.info('Final Accuracy: %s', sess.run(accuracy))

            #Once all the training has been done, save the log files and checkpoint model
            logging.info('Finished training! Saving model to disk now.')
            # saver.save(sess, "./flowers_model.ckpt")
            sv.saver.save(sess, sv.save_path, global_step = sv.global_step)


if __name__ == '__main__':
    run()

                

