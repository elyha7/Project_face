from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
# class NullDevice():
#     def write(self, s):
#         pass
# original_stdout = sys.stdout
# sys.stdout = NullDevice()
# sys.stderr = NullDevice()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import dlib
from skimage import io
import subprocess
from datetime import datetime
import math
import time
from data import inputs
import numpy as np
from model import select_model, get_checkpoint
from utils import ImageCoder, make_batch
from detect import face_detection_model
import json
import csv
import scipy.misc
from myfunctions import get_faces,save_pictures,get_faces_frame
import cv2
from flask import Flask, render_template, Response
from visual_api import mark_on_image,add_info_field,Face_info,generate_colors,make_classes
import tensorflow as tf
#TIMER=0
app = Flask(__name__)
RESIZE_FINAL = 227
GENDER_LIST =['M','F']
AGE_LIST = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
TIMER=0

filename='names.txt'
model_type= 'inception'
class_type1= 'age'
class_type2= 'gender'
model_dir1= '/home/elyha7/programms/docker_work/checkpoint_age'
model_dir2= '/home/elyha7/programms/docker_work/checkpoint_gender'
target1='/home/elyha7/programms/docker_work/age.csv'
target2='/home/elyha7/programms/docker_work/gender.csv'
app = Flask(__name__)
class Face_info(object):
    def __init__(self,age=None,gender=None,img=None,cords=None,time=None,name=None):
        self.name = name
        self.age = age
        self.gender = gender
        self.img = img
        self.time = time
        self.cords = cords
def gen_frame(capture):
    #here is the source of video set
    cap = cv2.VideoCapture(capture)
    colors=generate_colors()
    while True:
        success, frame = cap.read()
        global TIMER
        TIMER+=1
        if TIMER%10==0:
            # if timer%10!=0:
            #     yield()
            #place for procesing and marking
            print('\n1111111111111\n')
            #frame=np.array(frame,dtype=np.uint8)
            images,Faces = get_faces_frame(frame)#<--your func
            print('\n%s\n'%len(Faces))
            if len(Faces)==0:
                success, jpeg = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            classes =make_classes(images,Faces)
            names=save_pictures(classes)
            # for i in range(len(classes)):
            #     classes[i].gender='F'
            #     classes[i].age='12'
            frame=mark_on_image(frame,classes,colors)
            construct(filename,class_type1,model_type,model_dir1,target=target1,classes=classes)
            construct(filename,class_type2,model_type,model_dir2,target=target2,classes=classes)
            frame = add_info_field(frame,classes,colors)
            # #print("here")
            # for i in range(len(images)):
            #     print(classes[i].name,classes[i].age,classes[i].gender)
            # colors=generate_colors()
            # frame= mark_on_image(frame,Faces,colors)
            # frame = add_info_field(frame,Faces,colors)
            assert success, 'Wrong frame'

            success, jpeg = cv2.imencode('.jpg', frame)

            assert success, 'Encode error'
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
    

@app.route('/')
def page():
    return Response(gen_frame(0), mimetype='multipart/x-mixed-replace; boundary=frame')


#if __name__ == '__main__':
#    app.run(port =5002,debug=True) 


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
    #print('Guess @ 1 %s, prob = %.2f' % best_choice)
    
    nlabels = len(label_list)
    if nlabels > 2:
        output[best] = 0
        second_best = np.argmax(output)

        #print('Guess @ 2 %s, prob = %.2f' % (label_list[second_best], output[second_best]))
    return best_choice
         
def batchlist(srcfile):
    with open(srcfile, 'r') as csvfile:
        reader = csv.reader(csvfile)
        if srcfile.endswith('.csv') or srcfile.endswith('.tsv'):
            #print('skipping header')
            reader.next()
        
        return [row[0] for row in reader]

def construct(filename,class_type,model_type,model_dir,checkpoint='checkpoint', device='/cpu:0',target=None,classes=None):  # pylint: disable=unused-argument
    # sys.stdout = os.devnull
    # sys.stderr = os.devnull
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
            # sys.stdout = sys.__stdout__
            # sys.stderr = sys.__stderr__
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
   
            for it,f in enumerate(files):
                image_file = resolve_file(f)
            
                if image_file is None: continue

                try:
                    best_choice = classify(sess, label_list, softmax_output, coder, images, image_file)
                    #results[it][0]=f
                    #print('f=%s\nresult='%f,results)
                    if writer is not None:
                        writer.writerow((f, best_choice[0], '%.2f' % best_choice[1]))
                        #print("\nClass_type=",class_type)
                        if class_type=='age':
                            #print("\n%s\n"%it)
                            classes[it].name=f
                            classes[it].age=best_choice[0]
                            # print(best_choice[0],'\n')
                            # print(results,'\n')
                        if class_type=='gender':
                            #print("\n222222222\n")
                           classes[it].gender=best_choice[0]
                except Exception as e:
                    print(e)
                    print('Failed to run image %s ' % image_file)
                it+=1;    
            if output is not None:
                output.close()
            # print(results)
            #sess.close()
            #print('\n!!!!!!!!!!!!!!!\n')
def make_result_list(number):
    result=[]
    res=[0,0,0]
    for i in range(number):
        result.append(res)
    return result       
def main():
    app.run(port =5002,debug=True) 
    # filename='../selfi1.jpg'
    # images,boxes=get_faces(filename,0)
    # #print(boxes)
    # names=save_pictures(images,dest='../samples')
    # #print(names,boxes)
    # classes=make_classes(images,boxes)
    # #result= np.zeros(shape=(len(test_list),3))
    # #result=list(result)
    # #print(result)
    # filename1='names.txt'
    # class_type1= 'age'
    # model_type1= 'inception'
    # model_dir1= '/home/elyha7/programms/docker_work/checkpoint_age'
    # target1='/home/elyha7/programms/docker_work/age.csv'
    # construct(filename1,class_type1,model_type1,model_dir1,target=target1,classes=classes)
    
    # filename2='names.txt'
    # class_type2= 'gender'
    # model_type2= 'inception'
    # model_dir2= '/home/elyha7/programms/docker_work/checkpoint_gender'
    # target2='/home/elyha7/programms/docker_work/gender.csv'
    # construct(filename2,class_type2,model_type2,model_dir2,target=target2,classes=classes)
    # #print("here")
    # for i in range(len(images)):
    #     print(classes[i].name,classes[i].age,classes[i].gender)
    #construct(filename1,class_type1,model_type1,model_dir1,target=target1)
if __name__ == '__main__':
    main()

