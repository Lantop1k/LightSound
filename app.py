import os
import time
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from scipy.io.wavfile import write as write_wav
from flask import Flask, render_template, request, jsonify, send_from_directory
from pydub import AudioSegment
from dash import Dash, html

app = Flask(__name__)
app.secret_key = 'Lantop2333'

# Dash (just to avoid error)
dash_app = Dash(__name__, server=app, url_base_pathname='/dash/')
dash_app.layout = html.Div([html.H3("Piano Ready!")])

# ===========================
# SETTINGS
# ===========================
OUTPUT_DIR = "/tmp"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SAMPLE_RATE = 44100
DURATION_PER_STEP = 120 / 1000  # 120ms = natural piano feel

# ===========================
# YOUR 51 REAL SAMPLES: A1.ogg → A51.ogg
# ===========================
PIANO_SAMPLES = {}
COLOR_LIST = [
    (139, 0, 0),      # 1 → A1.ogg (dark red)
    (255, 69, 0),
    (204, 204, 0),
    (102, 152, 0),
    (0, 100, 0),      # 5
    (0, 50, 69),
    (0, 0, 139),
    (75, 0, 130),
    (112, 0, 171),
    (148, 0, 211),    # 10
    (157, 0, 106),
    (165, 0, 0),
    (210, 0, 128),
    (255, 94, 0),
    (221, 221, 0),    # 15
    (111, 175, 0),
    (0, 128, 0),
    (0, 64, 85),
    (0, 0, 170),
    (92, 0, 159),     # 20
    (119, 0, 96),
    (159, 0, 226),
    (175, 0, 113),
    (191, 0, 0),
    (223, 59, 128),   # 25
    (255, 119, 0),
    (238, 238, 0),
    (119, 159, 0),
    (0, 160, 0),
    (0, 80, 100),     # 30
    (0, 0, 200),
    (109, 0, 188),
    (140, 0, 215),
    (170, 0, 241),
    (194, 0, 121),    # 35
    (217, 0, 0),
    (236, 72, 0),
    (255, 144, 0),
    (250, 250, 0),
    (128, 224, 0),    # 40
    (0, 192, 0),
    (0, 96, 115),
    (0, 0, 230),
    (126, 0, 217),
    (159, 26, 236),   # 45
    (191, 51, 255),
    (217, 26, 128),
    (243, 0, 0),
    (249, 85, 0),
    (255, 169, 0),    # 50
    (255, 255, 51),   # 51 → A51.ogg (bright yellow)
]

# Find this function "load_samples()" and replace the whole thing with this:
def load_samples():
    count = 0
    sample_folder = os.path.join(os.path.dirname(__file__), "samples")
    for i in range(1, 52):
        path = os.path.join(sample_folder, f"A{i}.ogg")
        if not os.path.exists(path):
            print(f"Missing A{i}.ogg")
            continue
        try:
            audio = AudioSegment.from_ogg(path).set_frame_rate(SAMPLE_RATE).set_channels(1)
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            samples /= 32768.0
            PIANO_SAMPLES[i] = samples
            count += 1
        except Exception as e:
            print(f"Error A{i}.ogg: {e}")
    print(f"Loaded {count}/51 piano samples")

load_samples()
# ===========================
# Find closest color (with tolerance)
# ===========================
def find_closest_color(r, g, b):
    target = np.array([r, g, b])
    best_idx = 1
    best_dist = float('inf')
    for idx, color in enumerate(COLOR_LIST, 1):
        dist = np.linalg.norm(target - np.array(color))
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
        if dist < 30:  # very close → early exit
            return idx
    return best_idx

# ===========================
# Generate real piano sound
# ===========================
def generate_tone(sample_indices, duration=DURATION_PER_STEP):
    target_len = int(SAMPLE_RATE * duration)
    mixed = np.zeros(target_len)

    for idx in set(sample_indices):
        if idx not in PIANO_SAMPLES:
            continue
        sample = PIANO_SAMPLES[idx]
        if len(sample) >= target_len:
            seg = sample[:target_len]
        else:
            seg = np.concatenate([sample, np.zeros(target_len - len(sample))])
        mixed += seg

    if np.max(np.abs(mixed)) > 0:
        mixed /= np.max(np.abs(mixed))  # normalize

    # soft fade out
    fade = min(1000, target_len//6)
    if fade > 0:
        mixed[-fade:] *= np.linspace(1, 0, fade)

    return mixed

# ===========================
# ROUTES
# ===========================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/drawing2audio')
def drawing2audio():
    return render_template('drawing_to_note.html')

@app.route('/submit', methods=['POST'])
def submit():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "No image"}), 400

    img_data = data['image'].split(',')[1]
    img = Image.open(BytesIO(base64.b64decode(img_data))).convert('RGBA')
    pixels = img.load()
    w, h = img.size

    notes = []  # list of (color_set, width_fraction)
    current_color_set = set()
    note_start_x = 0

    print("\nReading your drawing...")

    for x in range(w + 1):
        this_column_colors = set()
        if x < w:
            for y in range(h):
                r, g, b, a = pixels[x, y]
                if a < 100 or (max(r, g, b) - min(r, g, b) < 30 and r < 80):
                    continue
                idx = find_closest_color(r, g, b)
                this_column_colors.add(idx)

        if this_column_colors == current_color_set and x < w:
            continue

        if current_color_set and note_start_x < x:
            width_px = x - note_start_x
            notes.append((current_color_set.copy(), width_px))
            print(f"→ Note {sorted(current_color_set)} over {width_px}px")

        note_start_x = x
        current_color_set = this_column_colors

    if not notes:
        return jsonify({"error": "No color detected"}), 400

    # Scale total duration (e.g., max 9s, or proportional but capped)
    TOTAL_DURATION = 9.0  # adjust as needed
    total_width = sum(width for _, width in notes)  # or just w if full
    segments = []
    for color_indices, width_px in notes:
        dur = (width_px / w) * TOTAL_DURATION  # fraction of width → time
        target_len = int(SAMPLE_RATE * dur)
        mixed = np.zeros(target_len, dtype=np.float32)

        for idx in color_indices:
            if idx not in PIANO_SAMPLES:
                continue
            sample = PIANO_SAMPLES[idx]
            if len(sample) > target_len:
                seg = sample[:target_len]
            else:
                seg = np.concatenate([sample, np.zeros(target_len - len(sample))])
            mixed += seg

        if np.max(np.abs(mixed)) > 0.001:
            mixed /= np.max(np.abs(mixed))

        # Fade out
        fade = min(1200, target_len // 5)
        if fade > 0:
            mixed[-fade:] *= np.linspace(1, 0, fade)

        segments.append(mixed)

    final_audio = np.concatenate(segments)
    final_audio = np.clip(final_audio, -1.0, 1.0)
    audio_i16 = np.int16(final_audio * 32767)

    filename = f"piano_{int(time.time()*1000)}.wav"
    write_wav(os.path.join(OUTPUT_DIR, filename), SAMPLE_RATE, audio_i16)

    total_sec = len(final_audio) / SAMPLE_RATE
    print(f"SUCCESS! {len(notes)} note(s), {total_sec:.2f} seconds → {filename}\n")

        return jsonify({"url": f"/audio/{filename}"})


# Change the last route (the one that serves audio) to this:
@app.route('/audio/<path:filename>')
def serve_audio(filename):
    return send_from_directory("/tmp", filename)

# ===========================
# RUN
# ===========================
if __name__ == '__main__':
    app.run()
