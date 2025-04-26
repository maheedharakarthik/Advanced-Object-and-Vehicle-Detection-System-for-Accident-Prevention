from flask import Flask, request, jsonify, send_file
import RPi.GPIO as GPIO
import subprocess
import io
from picamera import PiCamera
from time import sleep

app = Flask(__name__)

# Setup GPIO
LED_PINS = [8, 10, 11]
GPIO.setmode(GPIO.BOARD)
for pin in LED_PINS:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

# Initialize PiCamera
camera = PiCamera()

@app.route('/image')
def stream_image():
    stream = io.BytesIO()
    camera.capture(stream, format='jpeg')
    stream.seek(0)
    return send_file(stream, mimetype='image/jpeg')

@app.route('/change_status', methods=['POST'])
def change_status():
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    for pin in LED_PINS:
        pin_str = str(pin)
        if pin_str in data:
            GPIO.output(pin, GPIO.HIGH if data[pin_str] else GPIO.LOW)
    return jsonify({"status": "LEDs updated"})

@app.route('/speak', methods=['POST'])
def speak():
    data = request.json
    speech_text = data.get('text')
    if not speech_text:
        return jsonify({"error": "No text provided"}), 400
    
    try:
        subprocess.run(["espeak", speech_text])
        return jsonify({"status": "Speech executed"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000)
    finally:
        GPIO.cleanup()
