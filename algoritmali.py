import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmenter = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

background_image = None
background_selected = False 

def select_background():
    global background_image, background_selected
    filename = filedialog.askopenfilename(initialdir="/", title="Arka Plan Seç",
                                          filetypes=(("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")))
    print("Seçilen dosya:", filename)
    if filename:
        background_image = cv2.imread(filename)
        if background_image is not None:
            background_image = cv2.resize(background_image, (640, 480))  
            if not background_selected: 
                print("Arka plan başarıyla yüklendi!")
                background_selected = True 
        else:
            print("Arka plan yüklenemedi, dosya formatını kontrol edin.")
    else:
        print("Dosya seçilmedi!")


root = tk.Tk()
root.title("Arka Plan Seçimi")

btn_select = tk.Button(root, text="Arka Plan Seç", command=select_background)
btn_select.pack()

cap = cv2.VideoCapture(0)

def process_frame():
    global background_image
    ret, frame = cap.read()
    
    if not ret:
        return

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = segmenter.process(image_rgb)
    mask = result.segmentation_mask > 0.5

    # Yumuşatma ve morfolojik işlemler
    kernel = np.ones((5, 5), np.uint8)  # Maske için çekirdek
    mask = cv2.dilate(mask.astype('uint8'), kernel, iterations=1)  # Kenar genişletme
    mask = cv2.GaussianBlur(mask, (15, 15), 0)  # Kenarları bulanıklaştırma

    # Maskeyi tersine çeviriyoruz (arka plan ve ön planı ayırmak için)
    inverted_mask = np.where(mask > 0, 0, 1).astype('uint8')  # Ters maske: Arka planı tespit ediyor
    
    # Eğer bir arka plan seçildiyse, bu arka planı uyguluyoruz
    if background_image is not None:
        background = background_image.copy()
    else:
        background = frame.copy()  

    # ** Boyut eşitleme işlemi **:
    if background.shape != frame.shape:  
        background = cv2.resize(background, (frame.shape[1], frame.shape[0]))  # Çerçeve boyutlarına eşitle
    
    # Maskeyi kullanarak yeni arka planı ve kişiyi birleştiriyoruz
    foreground = cv2.bitwise_and(frame, frame, mask=mask.astype('uint8'))  # Ön plan (kişi)
    background_applied = cv2.bitwise_and(background, background, mask=inverted_mask.astype('uint8'))  # Arka plan

    # Ön plan ve arka planı birleştiriyoruz
    combined_frame = cv2.addWeighted(foreground, 1.0, background_applied, 1.0, 0)  # Opaklık ayarı
    
    # Görüntüyü Tkinter penceresinde gösterme
    frame_rgb = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    lbl_video.imgtk = imgtk
    lbl_video.configure(image=imgtk)

    # Döngüyü devam ettiriyoruz
    lbl_video.after(10, process_frame)

# Tkinter'da video için etiket (label) oluşturma
lbl_video = tk.Label(root)
lbl_video.pack()

process_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
