from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
sys.path.append('/home/elyha7/programms/docker_work/rude_carnie/all_func')
import dlib
from skimage import io
#import os
import subprocess
from datetime import datetime
import math
import time
from data import inputs
import numpy as np
import tensorflow as tf
from model import select_model, get_checkpoint
from utils import ImageCoder, make_batch
from detect import face_detection_model
import os
import json
import csv
import scipy.misc
from myfunctions import get_faces,save_pictures
RESIZE_FINAL = 227
GENDER_LIST =['M','F']
AGE_LIST = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']

# tf.app.flags.DEFINE_string('model_dir', '',
#                            'Model directory (where training data lives)')

# tf.app.flags.DEFINE_string('class_type', 'age',
#                            'Classification type (age|gender)')


# tf.app.flags.DEFINE_string('device_id', '/cpu:0',
#                            'What processing unit to execute inference on')

# tf.app.flags.DEFINE_string('filename', '',
#                            'File (Image) or File list (Text/No header TSV) to process')

# tf.app.flags.DEFINE_string('target', '',
#                            'CSV file containing the filename processed along with best guess and score')

# tf.app.flags.DEFINE_string('checkpoint', 'checkpoint',
#                           'Checkpoint basename')

# tf.app.flags.DEFINE_string('model_type', 'default',
#                            'Type of convnet')

# tf.app.flags.DEFINE_string('requested_step', '', 'Within the model directory, a requested step to restore e.g., 9000')

# tf.app.flags.DEFINE_boolean('single_look', False, 'single look at the image or multiple crops')

# tf.app.flags.DEFINE_string('face_detection_model', '', 'Do frontal face detection with model specified')

# tf.app.flags.DEFINE_string('face_detection_type', 'cascade', 'Face detection model type (yolo_tiny|cascade)')

# FLAGS = tf.app.flags.FLAGS

def one_of(fname, types):
    for ty in types:
        if fname.endswith('.' + ty):
            return True
    return False

def resolve_file(fname):
    if os.path.exists(fname): return fname
    for suffix in ('.jpg', '.png', '.JPG', '.PNG', '.jpeg'):
        cand = fname + suffix
        if os.path.exists(cand):
            return cand
    return None

def classify(sess, label_list, softmax_output, coder, images, image_file):

    #print('Running file %s' % image_file)
    image_batch = make_batch(image_file, coder, False)
    batch_results = sess.run(softmax_output, feed_dict={images:image_batch.eval()})
    output = batch_results[0]
    batch_sz = batch_results.shape[0]
    for i in range(1, batch_sz):
        output = output + batch_results[i]
        
    output /= batch_sz
    best = np.argmax(output)
    best_choice = (label_list[best], output[best])
    print('Guess @ 1 %s, prob = %.2f' % best_choice)
    
    nlabels = len(label_list)
    if nlabels > 2:
        output[best] = 0
        second_best = np.argmax(output)

        print('Guess @ 2 %s, prob = %.2f' % (label_list[second_best], output[second_best]))
    return best_choice
         
def batchlist(srcfile):
    with open(srcfile, 'r') as csvfile:
        reader = csv.reader(csvfile)
        if srcfile.endswith('.csv') or srcfile.endswith('.tsv'):
            #print('skipping header')
            reader.next()
        
        return [row[0] for row in reader]

def construct(filename,class_type,model_type,model_dir,checkpoint='checkpoint', device='/cpu:0',target=None):  # pylint: disable=unused-argument

    files = []
    with tf.Graph().as_default():
        sess = tf.InteractiveSession()
    #with tf.Session() as sess:
        #print('\n111111111111111\n')
        #tf.reset_default_graph()
        label_list = AGE_LIST if class_type == 'age' else GENDER_LIST
        nlabels = len(label_list)
        #print('\n222222222222222\n')
        #print('Executing on %s' % FLAGS.device_id)
        model_fn = select_model(model_type)

        with tf.device(device):
            
            images = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])
            logits = model_fn(nlabels, images, 1, False)
            init = tf.global_variables_initializer()
            #print('\n333333333333333\n')
            requested_step = None
        
            checkpoint_path = '%s' % (model_dir)

            model_checkpoint_path, global_step = get_checkpoint(checkpoint_path, requested_step, checkpoint)
            #print("\nglobal_step=",global_step)
            saver = tf.train.Saver()
            #print('\n44444444444444444\n')
            #print("PATH=",model_checkpoint_path,'\n')
            saver.restore(sess, model_checkpoint_path)
            #print('\n55555555555555555\n')            
            softmax_output = tf.nn.softmax(logits)

            coder = ImageCoder()

            # Support a batch mode if no face detection model
            if len(files) == 0:
                files.append(filename)
                # If it happens to be a list file, read the list and clobber the files
                if one_of(filename, ('csv', 'tsv', 'txt')):
                    files = batchlist(filename)

            writer = None
            output = None
            if target:
                #print('Creating output file %s' % FLAGS.target)
                output = open(target, 'w')
                writer = csv.writer(output)
                writer.writerow(('file', 'label', 'score'))


            for f in files:
                image_file = resolve_file(f)
            
                if image_file is None: continue

                try:
                    best_choice = classify(sess, label_list, softmax_output, coder, images, image_file)
                    if writer is not None:
                        writer.writerow((f, best_choice[0], '%.2f' % best_choice[1]))
                except Exception as e:
                    print(e)
                    print('Failed to run image %s ' % image_file)

            if output is not None:
                output.close()
            sess.close()
            #print('\n!!!!!!!!!!!!!!!\n')
        
def main():
    filename='../selfi.jpg'
    test_list,boxes=get_faces(filename,0)
    names=save_pictures(filename,test_list,dest='../samples')
    print(names,boxes)
    filename1='names.txt'
    class_type1= 'age'
    model_type1= 'inception'
    model_dir1= '/home/elyha7/programms/docker_work/checkpoint_age'
    target1='/home/elyha7/programms/docker_work/age.csv'
    construct(filename1,class_type1,model_type1,model_dir1,target=target1)
    
    filename2='names.txt'
    class_type2= 'gender'
    model_type2= 'inception'
    model_dir2= '/home/elyha7/programms/docker_work/checkpoint_gender'
    target2='/home/elyha7/programms/docker_work/gender.csv'
    construct(filename2,class_type2,model_type2,model_dir2,target=target2)
    #construct(filename1,class_type1,model_type1,model_dir1,target=target1)
if __name__ == '__main__':
    main()

