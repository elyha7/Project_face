import cv2
from flask import Flask, render_template, Response
from visual_api import mark_on_image,add_info_field,Face_info


app = Flask(__name__)


def gen_frame(capture):
	#here is the source of video set
    cap = cv2.VideoCapture(capture)
    
    while True:
        success, frame = cap.read()
        #place for procesing and marking
        """
        Faces = get_faces(frame)#<--your func
        frame,colors = mark_on_image(frame,Faces)
        frame = add_info_field(frame,Faces,colors)
        """
        assert success, 'Wrong frame'

        success, jpeg = cv2.imencode('.jpg', frame)

        assert success, 'Encode error'

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
    

@app.route('/')
def page():
    return Response(gen_frame(0), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(port =5002,debug=True) 
