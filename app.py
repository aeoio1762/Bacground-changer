from flask import Flask, request, render_template
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import os
from flask import Flask, request, render_template, send_file


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/arkaplan-degistirme', methods=['POST'])
def arkaplan_degistirme():
    # Gelen isteğin içeriğini al
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    # Arka plan görüntüsünü oluştur
    _, img_encoded = cv2.imencode('.png', image)
    nparr = np.frombuffer(img_encoded, np.uint8)
    arkaplan = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Cisim görüntüsünü oluştur
    file_cisim = request.files['cisim']
    image_cisim = cv2.imdecode(np.frombuffer(file_cisim.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    # Cismin boyutlarını al
    cisim_yukseklik, cisim_genislik, _ = image_cisim.shape

    # Cisim görüntüsünü arkaplan boyutlarına uyacak şekilde yeniden boyutlandır
    cisim = cv2.resize(image_cisim, (arkaplan.shape[1], arkaplan.shape[0]))

    # Cisim alfa kanalını normalleştir
    alpha = cisim[:, :, 3] / 255.0

    # Cisim piksellerini transparan yap
    arkaplan[:, :, :3] = arkaplan[:, :, :3] * (1 - alpha)[:, :, np.newaxis]

    # Cisim piksellerini arkaplana ekle
    arkaplan[:, :, :3] = arkaplan[:, :, :3] + (cisim[:, :, :3] * alpha[:, :, np.newaxis])

    # Eksik bölgeleri tamamlamak için morfolojik işlemler uygula
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    arkaplan = cv2.morphologyEx(arkaplan, cv2.MORPH_CLOSE, kernel)

    # Çıktı dosyasını kaydet
    output_path = "static/output.png"
    cv2.imwrite(output_path, arkaplan)

    # Sonuç görüntüsünü döndür
    return send_file(output_path, mimetype='image/png')



@app.route('/arkaplan-kesme', methods=['GET', 'POST'])
def upload_and_remove():
    if request.method == 'POST':
        # Dosya yüklemesini al
        file = request.files['file']
        
        # Yüklü dosyayı geçici bir dosyaya kaydet
        upload_path = "uploads/" + file.filename
        file.save(upload_path)
        
        # Arka planı kaldırma işlemini gerçekleştir
        image = Image.open(upload_path)
        remove_bgnd = remove(image)
        
        # Çıktıyı yeni bir dosyaya kaydet
        output_path = "static/" + file.filename
        remove_bgnd.save(output_path)
        
        # Geçici dosyayı sil
        os.remove(upload_path)
        
        # HTML sayfasına sonuçları aktar
        return render_template('kirpma.html', input_path=upload_path, output_path=output_path)
    
    return render_template('kirpma.html')
if __name__ == '__main__':
    app.run(debug=True)


