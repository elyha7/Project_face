import sys
import dlib
from skimage import io
from matplotlib import pyplot as plt
import numpy as np
import scipy.misc
def get_faces(filename,threshold=0.0):
    face_detector = dlib.get_frontal_face_detector()
    image = plt.imread(filename)
    detected_faces,scores,idx = face_detector.run(image, 1,threshold)
    faces=[]
    boxes=[]
    for i,j in enumerate(detected_faces):
        boxes.append(j)
        p_1=j.top()
        p_2=j.bottom()
        p_3=j.left()
        p_4=j.right()
        faces.append(image[p_1:p_2,p_3:p_4,:])
    return faces,boxes
def save_pictures(filename,faces,dest='./'):
    names=[]
    thefile = open('names.txt', 'w')
    try:
        b=filename.rfind('/')
    except:
        b=-1
    c=filename.find('.jpg')
    a=filename[b+1:c]
    #print(a)
    for k,i in enumerate(faces):
        newname=dest+'/'+a+'_'+str(k)+'.jpg'
        #print('newname=',newname)
        names.append(newname)
        thefile.write("%s\n" % newname)
        scipy.misc.imsave(newname, i)
    return names
def make_gen_string(filename,img_dir,destination):
    gender_string='python guess.py --class_type gender --model_type inception --model_dir /home/elyha7/programms/docker_work/checkpoint_gender'
    gender_string+=' --filename '+img_dir+'/'+filename
    gender_string+=' --target '+destination+'/'+'gender.csv'
    return gender_string
def make_age_string(filename,img_dir,destination):
    age_string='python guess.py --class_type age --model_type inception --model_dir /home/elyha7/programms/docker_work/chekpoint_age'
    age_string+=' --filename '+img_dir+'/'+filename
    age_string+=' --target '+destination+'/'+'age.csv'
    return age_string
def write_result(ifile,ifile1,destination):
    reader = csv.reader(ifile)
    output=open(destination+'/result.csv','a')
    writer = csv.writer(output)
    k=0
    for k,i in enumerate(reader):
        if k==1:
            save=i[1]
            #writer.writerow((i[0],i[1],i[2]))
        k+=1
    reader = csv.reader(ifile1)
    for k,i in enumerate(reader):
        if k==1:
            writer.writerow((i[0],i[1],save))
    output.close()
        
