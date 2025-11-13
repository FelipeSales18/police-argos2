from ultralytics import YOLO
import cv2
import easyocr
import os
import re

# ==========================
# CONFIGURAÇÕES
# ==========================

# Caminho do seu modelo YOLO treinado
MODEL_PATH = "placa-br-yolov11.pt"

# Caminho da imagem a ser testada
IMAGE_PATH = "carro_teste.jpg"

# Inicializa o leitor OCR (apenas uma vez)
print("[INFO] Inicializando EasyOCR...")
reader = easyocr.Reader(['pt', 'en'], gpu=False)  # Usando CPU

# ==========================
# FUNÇÃO PARA OCR LOCAL
# ==========================
def ocr_placa(image):
    """
    Realiza OCR na imagem da placa usando EasyOCR
    Retorna apenas letras e números (formato de placa brasileira)
    """
    try:
        # Pré-processamento da imagem
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Aumenta o contraste
        gray = cv2.equalizeHist(gray)
        
        # Redimensiona para melhorar OCR (altura mínima de 100px)
        height, width = gray.shape
        if height < 100:
            scale = 100 / height
            new_width = int(width * scale)
            gray = cv2.resize(gray, (new_width, 100), interpolation=cv2.INTER_CUBIC)
        
        # Aplica OCR
        result = reader.readtext(gray, detail=0, paragraph=False)
        
        if not result:
            return "SEM TEXTO"
        
        # Junta todos os textos encontrados
        texto_completo = ''.join(result)
        
        # Remove espaços e caracteres especiais
        texto_limpo = re.sub(r'[^A-Z0-9]', '', texto_completo.upper())
        
        # Formata no padrão brasileiro (ABC1234 ou ABC1D23)
        if len(texto_limpo) >= 7:
            # Tenta formato Mercosul (ABC1D23)
            if len(texto_limpo) == 7:
                return f"{texto_limpo[:3]}{texto_limpo[3]}{texto_limpo[4]}{texto_limpo[5:]}"
            # Formato antigo (ABC1234)
            else:
                return texto_limpo[:7]
        
        return texto_limpo if texto_limpo else "ILEGÍVEL"
        
    except Exception as e:
        print(f"[ERRO OCR] {e}")
        return "ERRO"

# ==========================
# DETECÇÃO DE PLACAS
# ==========================

def main():
    # Verifica se o modelo existe
    if not os.path.exists(MODEL_PATH):
        print(f"[ERRO] Modelo não encontrado: {MODEL_PATH}")
        print("[INFO] Você precisa do modelo YOLOv11 treinado para placas brasileiras")
        return
    
    # Verifica se a imagem existe
    if not os.path.exists(IMAGE_PATH):
        print(f"[ERRO] Imagem não encontrada: {IMAGE_PATH}")
        print("[INFO] Coloque uma imagem de teste na pasta backend/")
        return
    
    # Carrega modelo YOLO
    print("[INFO] Carregando modelo YOLOv11...")
    model = YOLO(MODEL_PATH)
    
    # Lê a imagem
    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        print(f"[ERRO] Não foi possível ler a imagem: {IMAGE_PATH}")
        return
    
    print(f"[INFO] Imagem carregada: {frame.shape[1]}x{frame.shape[0]} pixels")
    
    # Faz a detecção
    print("[INFO] Detectando placas na imagem...")
    results = model(frame, verbose=False, conf=0.5)  # confidence mínima de 50%
    
    if len(results[0].boxes) == 0:
        print("[AVISO] Nenhuma placa detectada na imagem!")
        output_path = "resultado_sem_placa.jpg"
        cv2.imwrite(output_path, frame)
        print(f"[INFO] Imagem salva em: {output_path}")
        return
    
    print(f"[INFO] {len(results[0].boxes)} placa(s) detectada(s)")
    
    # Lista para armazenar os resultados
    placas_detectadas = []
    
    # Percorre as detecções encontradas
    for i, box in enumerate(results[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        
        # Adiciona margem ao recorte para melhorar OCR
        margin = 5
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(frame.shape[1], x2 + margin)
        y2 = min(frame.shape[0], y2 + margin)
        
        placa_crop = frame[y1:y2, x1:x2]
        
        if placa_crop.size == 0:
            print(f"[AVISO] Recorte {i} está vazio, pulando...")
            continue
        
        # Salva o recorte
        crop_path = f"placa_{i}.jpg"
        cv2.imwrite(crop_path, placa_crop)
        print(f"[INFO] Recorte salvo: {crop_path}")
        
        # OCR LOCAL
        print(f"[INFO] Lendo texto da placa {i}...")
        texto = ocr_placa(placa_crop)
        print(f"[✅ RESULTADO] Placa {i}: {texto}")
        
        # Armazena o resultado
        placas_detectadas.append(texto)
        
        # Desenha caixa verde e texto na imagem original
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Fundo preto para o texto ficar mais legível
        text_size = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), 
                     (x1 + text_size[0], y1), (0, 0, 0), -1)
        
        # Texto branco
        cv2.putText(frame, texto, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Salva resultado final
    output_path = "resultado_final.jpg"
    cv2.imwrite(output_path, frame)
    print(f"\n{'='*60}")
    print(f"[✅ SUCESSO] Detecção concluída!")
    print(f"[📊] Total de placas detectadas: {len(placas_detectadas)}")
    print(f"[📄] Placas encontradas: {', '.join(placas_detectadas)}")
    print(f"[💾] Resultado salvo em: {output_path}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
