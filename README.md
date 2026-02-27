# 🎥 Sanal Arka Plan Uygulamaları (DeepLabV3 & MediaPipe)

Bu proje, görüntü işleme ve derin öğrenme tekniklerini kullanarak video akışında gerçek zamanlı arka plan değiştirme (Sanal Arka Plan) sağlayan iki farklı Python uygulamasını içermektedir.



## 🛠️ Kullanılan Teknolojiler
* **Python 3.x**
* **OpenCV:** Görüntü işleme ve kamera yönetimi.
* **DeepLabV3 (PyTorch):** Google'ın gelişmiş segmentasyon modeli.
* **MediaPipe:** Yüksek hızlı selfie segmentasyon çözümü.
* **Tkinter:** Kullanıcı dostu arayüz ve dosya seçimi.

## 📁 Modüller ve Çalışma Prensipleri

### 1. Derin Öğrenme Tabanlı Segmentasyon (`deeplab_arka_plan.py`)
* **Model:** `google/deeplabv3_mobilenet_v2_1.0_513` modeli kullanılır.
* **Hassasiyet:** Kişi (person) sınıfını (indeks 15) hassas bir şekilde ayırt eder.
* **Görsel Kalite:** Gaussian Blur ve blending (karıştırma) teknikleri ile kenar yumuşatma uygulanarak kişi ve yeni arka planın doğal görünmesi sağlanır.

### 2. İnteraktif MediaPipe Arayüzü (`mediapipe_arayuz.py`)
* **GUI (Arayüz):** Tkinter butonu aracılığıyla kullanıcı, uygulama çalışırken bilgisayarından dilediği `.jpg` veya `.png` dosyasını arka plan olarak seçebilir.
* **Performans:** MediaPipe Selfie Segmentation modeli sayesinde düşük gecikme süresi ile çalışır.
* **Dinamik Boyutlandırma:** Seçilen arka plan resmi, kamera çözünürlüğüne (640x480) göre otomatik olarak yeniden boyutlandırılır.



## 🚀 Kurulum ve Çalıştırma

1. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install opencv-python numpy torch transformers mediapipe pillow