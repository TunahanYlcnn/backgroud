import cv2
import numpy as np
import torch
from transformers import AutoModelForSemanticSegmentation, AutoFeatureExtractor

# Modeli ve feature extractor'ı yükle
model_name = "google/deeplabv3_mobilenet_v2_1.0_513"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForSemanticSegmentation.from_pretrained(model_name)

# Cihaz ayarı (GPU veya CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Arka plan resmi yüklenir
background = cv2.imread('C:/Users/tunahan/Desktop/background/arkaPlan.jpg')

# Kamera açılır
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü preprocess et
    inputs = feature_extractor(images=frame, return_tensors="pt").to(device)

    # Modelden tahmin al
    with torch.no_grad():
        outputs = model(**inputs)

    # Çıktıları al
    logits = outputs.logits
    predicted = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

    # Arka plan maskesi
    mask = predicted == 15  # 15, "person" sınıfının indeksi (DeepLabV3 için)
    
    # Maskeyi orijinal çerçeve boyutuna yeniden boyutlandır
    mask_resized = cv2.resize(mask.astype(np.float32), (frame.shape[1], frame.shape[0]))

    # Arka planı çerçeve boyutuna göre ayarlayın
    background_resized = cv2.resize(background, (frame.shape[1], frame.shape[0]))

    # Kenar yumuşatma (Blur) uygulayarak daha yumuşak geçişler sağlanır
    mask_blurred = cv2.GaussianBlur(mask_resized, (21, 21), 0)

    # Maske yardımıyla pürüzsüz geçiş için blending yapılır
    frame_float = frame.astype(float)
    background_float = background_resized.astype(float)

    # Maskeyi üç kanala genişletme
    mask_blurred_3d = np.dstack([mask_blurred] * 3)

    # Daha yumuşak bir geçiş için arka plan ve ön planı birleştiriyoruz
    blended_frame = mask_blurred_3d * frame_float + (1 - mask_blurred_3d) * background_float

    # Float tipini uint8'e çevirip, sonucu ekrana veriyoruz
    blended_frame = blended_frame.astype(np.uint8)

    # Yeni çerçeveyi göster
    cv2.imshow('Arka Plan Değiştirilmiş', blended_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
