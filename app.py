from flask import Flask, render_template, request, url_for, abort
from PIL import Image, UnidentifiedImageError
import numpy as np
import os
from math import isfinite
from datetime import datetime
from utils.color_conversion import rgb_to_lab
from utils.color2gray_impl import color2gray_vanilla, color2gray_enhanced
from utils.guided_filter import guided_filter
from scipy.ndimage import gaussian_filter, uniform_filter, minimum_filter, median_filter, sobel

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    preview_url = None

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or not file.filename:
            abort(400, description="Please select an image.")
        if file and file.filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            original_path = os.path.join(UPLOAD_FOLDER, f"original_{timestamp}.jpg")
            try:
                img_pil = Image.open(file.stream).convert('RGB')
            except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
                abort(400, description="Invalid or unsupported image.")

            method = request.form.get('method', 'all')

            # Running all three methods on a large image is expensive — cap it lower.
            MAX_DIM = 800 if method == 'all' else 1024
            orig_w, orig_h = img_pil.size
            if max(orig_w, orig_h) > MAX_DIM:
                scale = MAX_DIM / max(orig_w, orig_h)
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
                img_pil_resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
                print(f"Downsampled from {orig_w}x{orig_h} to {new_w}x{new_h}")
            else:
                img_pil_resized = img_pil
                print(f"Processing at original size: {orig_w}x{orig_h}")

            img_rgb = np.array(img_pil_resized)

            try:
                theta = float(request.form.get('theta', 30))
                alpha = float(request.form.get('alpha', 4.0))
            except (ValueError, TypeError):
                abort(400, description="Invalid numeric parameter.")

            if (not all(map(isfinite, [theta, alpha]))
                    or not 0 <= theta <= 85
                    or alpha < 0):
                abort(400, description="Numeric parameter out of range.")

            img_pil.save(original_path)

            print("Converting to Lab...")
            lab = rgb_to_lab(img_rgb)

            if method in ('all', 'standard'):
                L = lab[:, :, 0]
                standard_gray = (L / 100.0 * 255).astype(np.uint8)
                standard_path = os.path.join(RESULT_FOLDER, f"standard_{timestamp}.jpg")
                Image.fromarray(standard_gray).save(standard_path)

            if method in ('all', 'vanilla'):
                print("Running vanilla Color2Gray...")
                vanilla_gray = color2gray_vanilla(img_rgb, theta=theta)
                vanilla_path = os.path.join(RESULT_FOLDER, f"vanilla_{timestamp}.jpg")
                Image.fromarray(vanilla_gray).save(vanilla_path)

            if method in ('all', 'enhanced'):
                print("Running enhanced Color2Gray...")
                enhanced_gray = color2gray_enhanced(
                    img_rgb,
                    theta=theta,
                    alpha=alpha,
                    guided_r_list=[3, 8, 16],
                    isolum_threshold=5.0,
                    lab_override=lab  # reuse Lab computed above
                )
                enhanced_path = os.path.join(RESULT_FOLDER, f"enhanced_{timestamp}.jpg")
                Image.fromarray(enhanced_gray).save(enhanced_path)

            results = {'original': f"uploads/original_{timestamp}.jpg"}
            if method in ('all', 'standard'):
                results['standard'] = f"results/standard_{timestamp}.jpg"
            if method in ('all', 'vanilla'):
                results['vanilla'] = f"results/vanilla_{timestamp}.jpg"
            if method in ('all', 'enhanced'):
                results['enhanced'] = f"results/enhanced_{timestamp}.jpg"

            preview_url = url_for('static', filename=results['original'])

    return render_template('index.html', results=results, preview_url=preview_url)

if __name__ == '__main__':
    print("Server starting: http://127.0.0.1:5001")
    app.run(host='127.0.0.1', port=5001, debug=True)
