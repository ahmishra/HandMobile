from flask import Flask, render_template, Response, url_for # Web utilities
from imutils.video import VideoStream # Better video
from camera import Camera # Importing camera module
import serial  # To communicate with Arduino

hardware_enabled = False  # If hardware is enabled
controller = "right" # Controlling hand
if hardware_enabled:
    ser = serial.Serial("COM6", 9600) # Serial object
else:
    ser = None

app = Flask(__name__)  # Initialzing the Flask app
camera = VideoStream(usePiCamera=False).start() # Initializing Webcam

@app.route("/")
def index():
    """
    Home Page:
    Where the main content is displayed
    """

    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """
    Video Feed
    The origin point of the video frames where `index` gets the data from
    """

    video = Camera(camera=camera, controller=controller, hardware_enabled=hardware_enabled, ser=ser)
    return Response(video.mainloop(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/credits")
def credits():
    """
    Credits Page:
    The grandiloquent credits page :)
    """

    return render_template("credits.html")


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False, host="0.0.0.0", port="5000")
