#!/usr/bin/env python3
"""
Extractor automático de frames de monedas desde video (OpenCV)
Uso:
  python3 capturar_desde_video.py --video video_cara.mp4 --salida dataset/cara --prefijo cara --cantidad 60
"""
import os
import cv2
import argparse

def extraer_monedas(video_path, output_dir, prefix, max_frames=60, img_size=256):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // max_frames)
    guardadas = 0
    frame_idx = 0

    while cap.isOpened() and guardadas < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            h, w, _ = frame.shape
            lado = min(h, w)
            y0, x0 = (h - lado) // 2, (w - lado) // 2
            recorte = frame[y0:y0+lado, x0:x0+lado]
            redim = cv2.resize(recorte, (img_size, img_size), interpolation=cv2.INTER_AREA)
            guardadas += 1
            out_file = os.path.join(output_dir, f"{prefix}_{guardadas:03d}.jpg")
            cv2.imwrite(out_file, redim, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        frame_idx += 1
    cap.release()
    print(f"[OK] Extraídas {guardadas} imágenes en {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Ruta al archivo .mp4 grabado con el teléfono")
    parser.add_argument("--salida", required=True, help="Carpeta destino (ej. dataset/cara)")
    parser.add_argument("--prefijo", default="moneda", help="Prefijo de los archivos")
    parser.add_argument("--cantidad", type=int, default=60, help="Cantidad de imágenes a extraer")
    args = parser.parse_args()
    extraer_monedas(args.video, args.salida, args.prefijo, args.cantidad)
