from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import mediapipe as mp

app = Flask(__name__)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=True)
mpDraw = mp.solutions.drawing_utils

@app.route("/process", methods=["POST"])
def process_image():
    try:
        data = request.json
        image_data = data.get("image")
        
        if not image_data or not image_data.startswith("data:image"):
            return jsonify({"error": "Formato de imagen no válido"}), 400

        # Decodificar la imagen
        img_data = base64.b64decode(image_data.split(",")[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Procesamiento de la imagen
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

        # Codificar la imagen procesada
        _, buffer = cv2.imencode(".jpg", frame)
        processed_img = base64.b64encode(buffer).decode("utf-8")

        return jsonify({"processed_frame": f"data:image/jpeg;base64,{processed_img}"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
