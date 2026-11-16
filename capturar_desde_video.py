#!/usr/bin/env python3
"""
Extractor automático de fotogramas de monedas desde video (OpenCV)
con detección circular inteligente (Hough Transform) y recorte centrado (Auto-Crop).
Optimiza la resolución útil de la moneda al 80-90% del encuadre para evitar
pérdida de nitidez al redimensionar a 256x256.

Uso:
  python3 capturar_desde_video.py --video dataset/cara.mp4 --salida dataset/cara --prefijo cara --cantidad 50
"""
import os
import cv2
import argparse
import numpy as np

def crop_coin_from_frame(frame, margin_factor=1.20, img_size=256, last_known_roi=None):
    scale = 0.25
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=80,
        param1=50,
        param2=30,
        minRadius=35,
        maxRadius=170
    )
    
    img_center = np.array([small.shape[1] / 2, small.shape[0] / 2])
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        best_c = min(circles, key=lambda c: np.linalg.norm(np.array([c[0], c[1]]) - img_center))
        cx = int(best_c[0] / scale)
        cy = int(best_c[1] / scale)
        r = int(best_c[2] / scale)
        last_roi = (cx, cy, r)
    elif last_known_roi is not None:
        cx, cy, r = last_known_roi
        last_roi = last_known_roi
    else:
        cx, cy, r = w // 2, h // 2, int(min(w, h) * 0.18)
        last_roi = (cx, cy, r)
        
    half_side = int(r * margin_factor)
    y1 = max(0, cy - half_side)
    y2 = min(h, cy + half_side)
    x1 = max(0, cx - half_side)
    x2 = min(w, cx + half_side)
    
    crop = frame[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    lado = min(ch, cw)
    crop = crop[:lado, :lado]
    
    resized = cv2.resize(crop, (img_size, img_size), interpolation=cv2.INTER_LANCZOS4)
    return resized, last_roi

def extraer_monedas(video_path, output_dir, prefix, max_frames=50, img_size=256):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Procesando {video_path} ({total_frames} frames totales)...")
    
    step = max(1, total_frames // (max_frames + 5))
    guardadas = 0
    frame_idx = 0
    last_roi = None

    while cap.isOpened() and guardadas < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0 and frame_idx > 10:
            recorte, last_roi = crop_coin_from_frame(frame, margin_factor=1.20, img_size=img_size, last_known_roi=last_roi)
            guardadas += 1
            out_file = os.path.join(output_dir, f"{prefix}_{guardadas:03d}.jpg")
            cv2.imwrite(out_file, recorte, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if guardadas % 10 == 0 or guardadas == max_frames:
                print(f"  -> {guardadas}/{max_frames} imágenes guardadas...")
        frame_idx += 1
        
    cap.release()
    print(f"[OK] Extraídas {guardadas} imágenes en {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Ruta al archivo .mp4 grabado")
    parser.add_argument("--salida", required=True, help="Carpeta destino (ej. dataset/cara)")
    parser.add_argument("--prefijo", default="moneda", help="Prefijo de los archivos")
    parser.add_argument("--cantidad", type=int, default=50, help="Cantidad de imágenes a extraer")
    parser.add_argument("--size", type=int, default=256, help="Tamaño cuadrado de salida")
    args = parser.parse_args()
    extraer_monedas(args.video, args.salida, args.prefijo, args.cantidad, args.size)
