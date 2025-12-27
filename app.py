from flask import Flask, request, jsonify, send_file, render_template, send_file
import os
import uuid
import cv2
import pandas as pd
from io import BytesIO
import base64

from utils import (
    read_video, save_video, measure_distance,
    draw_player_stats, convert_pixel_distance_to_meters
)
import constants
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
from copy import deepcopy

app = Flask(__name__)

def extract_last_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    last_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        last_frame = frame
    cap.release()
    return last_frame


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze-video", methods=["POST"])
def analyze_video():

    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video_file = request.files["video"]
    uid = str(uuid.uuid4())

    input_path = f"input_videos/{uid}.mp4"
    output_path = f"output_videos/{uid}.avi"

    video_file.save(input_path)

    from main import main as process_video
    process_video(input_path, output_path)

    # extract image frame
    final_frame = extract_last_frame(output_path)
    _, buffer = cv2.imencode(".jpg", final_frame)
    output_image_base64 = base64.b64encode(buffer).decode("utf-8")

    # return frontend video URL + image + stats
    return jsonify({
        "message": "Processing completed",
        "video_url": f"/download/{output_path}",
        "image_base64": output_image_base64,
        "stats": "OK"
    })


@app.route("/download/<path:filename>")
def download(filename):
    return send_file(filename, as_attachment=False)


if __name__ == "__main__":
    app.run(debug=True)

