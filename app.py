"""
AdaIN Neural Style Transfer — Flask Application
================================================
Serves the web frontend and handles style-transfer inference requests.

Routes:
    GET  /           → Render index.html
    POST /transfer   → Accept content + style images, run AdaIN, return output URL
    GET  /output/<f> → Serve generated output images
"""
import io
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from flask import Flask, render_template, request, jsonify, send_from_directory,send_file

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization

# ── Configuration ──
BASE_DIR    = Path(__file__).resolve().parent
VGG_PATH    = BASE_DIR / 'vgg_normalized.pth'
DECODER_PATH = BASE_DIR / 'decoder_final.pth'

# ── Flask App ──
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20 MB limit

# ── Device ──
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Load Models (once at startup) ──
print(f'[INFO] Loading models on {device} ...')

encoder = VGGEncoder(str(VGG_PATH)).to(device)
encoder.eval()

decoder = Decoder().to(device)
state_dict = torch.load(str(DECODER_PATH), map_location=device, weights_only=True)
decoder.load_state_dict(state_dict)
decoder.eval()

print('[INFO] Models loaded successfully.')

# ── Image Transform ──
def load_image(image_file, size=512):
    """Load a PIL image, resize, and convert to tensor."""
    img = Image.open(image_file).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0)  # [1, 3, H, W]


# ── Routes ──
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/transfer', methods=['POST'])
def transfer():
    """Run AdaIN style transfer on uploaded content + style images."""
    # Validate files
    if 'content' not in request.files or 'style' not in request.files:
        return jsonify({'error': 'Both content and style images are required.'}), 400

    content_file = request.files['content']
    style_file   = request.files['style']

    if content_file.filename == '' or style_file.filename == '':
        return jsonify({'error': 'Please select valid image files.'}), 400

    alpha = float(request.form.get('alpha', 1.0))
    alpha = max(0.0, min(1.0, alpha))  # Clamp to [0, 1]

    try:
        # Load images as tensors
        content_tensor = load_image(content_file).to(device)
        style_tensor   = load_image(style_file).to(device)

        # Inference
        with torch.no_grad():
            content_feat = encoder(content_tensor, is_test=True)
            style_feat   = encoder(style_tensor, is_test=True)

            # AdaIN
            t = adaptive_instance_normalization(content_feat, style_feat)

            # Alpha blending: interpolate between content features and stylized features
            t = alpha * t + (1.0 - alpha) * content_feat

            # Decode
            output = decoder(t)
            output = output.clamp(0, 1)

        # Send image directly without saving
            buffer = io.BytesIO()

            save_image(output, buffer, format="PNG")
            buffer.seek(0)

        return send_file(
                buffer,
                mimetype="image/png",
                as_attachment=False,
                download_name="stylized_output.png"
            )

    except Exception as e:
        return jsonify({'error': f'Style transfer failed: {str(e)}'}), 500


# ── Main ──
if __name__ == '__main__':
    print('[INFO] Starting server at http://localhost:5000')
    app.run(debug=True, host='0.0.0.0', port=5000)
