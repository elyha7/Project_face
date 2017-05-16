import numpy as np
import cv2
import time
#Face structure class
#(just example how it may look you may use your class)
#(but so you will need to correct the code below)

#age int or float number
#gender str (example -male/female)
#img - cropped face img
#cords-tuple of two tuples- cords of upper lef corner and bottom right corner of bounding box
class Face_info(object):
    def __init__(self,age=None,gender=None,img=None,cords=None,time=None,name=None):
        self.name = name
        self.age = age
        self.gender = gender
        self.img = img
        self.time = time
        self.cords = cords

#func to draw bounding boxes for on the frame for a given faces
#Args:
#     img - numpy array(frame you want to draw on)
#     Faces - list of Face_info objects
#     color_rand -Bool if generate random colors for each bounding box
#     color = 3-tuple (R,G,B)(may be BGR) working when color_rand = True
#     thick - int number, thickness of box sides
def make_frame(bytes,stream):    
    bytes+=stream.read(1024)
    #print bytes
    a = bytes.find('\xff\xd8')
    b = bytes.find('\xff\xd9')
    if a!=-1 and b!=-1:
        jpg = bytes[a:b+2]
        bytes= bytes[b+2:]
        i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.CV_LOAD_IMAGE_COLOR)
        return i
    return 0   
def generate_colors():
    colors=[]
    for i in range(10):
        color = tuple(np.random.randint(0,256,size =3))
        colors.append(color)
    return colors
def mark_on_image(img,Faces,colors, thick = 2):
    #making copy of image to draw on and not affect the original
    img_to_ret = np.array(img)
    #put generated colors here
    #main cycle
    for i,Face in enumerate(Faces):
        print(colors[i])
        #support points to put rect(b. box) on
        pt1 = Face.cords[0][::-1]
        pt2 = Face.cords[1][::-1]
        #generate colors
        cv2.rectangle(img_to_ret,pt1,pt2,colors[i],thick)
    return img_to_ret
def add_info_field(img,faces,colors):
    #making canvas
    canv_shape = (img.shape[0],400,3)
    canvas = np.ones(canv_shape,dtype =np.uint8)*255
    
    #setting font and starting cord to draw from
    font = cv2.FONT_HERSHEY_PLAIN 
    pos = [10,10]
    #main cycle
    for face,color in zip(faces,colors):
        print (color,face.age)
        #sign point before text
        patch = np.ones((5,5,3),dtype =np.uint8)*color
        #constructing text
        age_text = "age:"+ str(face.age)
        gender_text = "gender:" +face.gender
        
        full_text = gender_text + "||"+ age_text
        
        #drawing
        #NOTE: cv2 cords system is inverted relative to numpy indexing.
        pt2 = (pos[1]-2,pos[0]-2)
        pt1 = (pos[1]+2,pos[0]+2)
        
        #cv2.rectangle(canvas,pt1,pt2,(0,255,255),thick)
        #print canvas[pos[0]-2:pos[0]+3,pos[1]-2:pos[1]+3,:].shape,patch.shape
        canvas[pos[1]-2:pos[1]+3,pos[0]-2:pos[0]+3,:] = patch
        cv2.putText(canvas,full_text,org =(pos[0]+5,pos[1]+2),fontFace =font,
                    fontScale =1,
                    color = color,
                    thickness=1)
         
        #moving position down           )
        pos[1]+=20
    #cv2.imshow('canv',canvas)
    #cv2.waitKey()

    #putting canvas and original together
    final_shape = (img.shape[0],img.shape[1]+canvas.shape[1],3)
    img_to_ret = np.zeros(final_shape,dtype = np.uint8)
    img_to_ret[:,0:img.shape[1]] = img
    img_to_ret[:,img.shape[1]:img.shape[1]+canvas.shape[1],:] = canvas
    #print img_to_ret.shape
    return img_to_ret
def make_classes(images,boxes):
    classes=[]
    for i in range(len(images)):
        a= Face_info(img=images[i],cords=boxes[i])
        a.time=time.ctime()
        classes.append(a)

    return classes