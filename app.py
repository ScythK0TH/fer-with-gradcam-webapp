# app.py
from flask import Flask, render_template, Response
from controllers.stream import generate_mjpeg, start_camera, stop_camera
import atexit

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    start_camera(0)
    # ensure camera released on exit
    atexit.register(stop_camera)
    app.run(host='0.0.0.0', port=3000, threaded=True)
