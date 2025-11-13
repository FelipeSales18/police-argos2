from ultralytics import YOLO
import cv2
import easyocr
import os
import re
import numpy as np

# ==========================
# CONFIGURAÇÕES
# ==========================

# Caminho do seu modelo YOLO treinado
MODEL_PATH = "placa-br-yolov11.pt"

# Caminho da imagem a ser testada
IMAGE_PATH = "carro_teste.jpg"

# Modo de depuração (salva imagens intermediárias)
DEBUG_MODE = True

# Habilita heatmap de confiança
ENABLE_HEATMAP = True

# Inicializa o leitor OCR (apenas uma vez)
print("[INFO] Inicializando EasyOCR...")
reader = easyocr.Reader(['pt', 'en'], gpu=False)  # Usando CPU

# ==========================
# FUNÇÃO DE PRÉ-PROCESSAMENTO
# ==========================
def pre_processar_imagem(frame):
    """
    Aplica técnicas de pré-processamento para melhorar a detecção
    """
    print("[INFO] Aplicando pré-processamento...")
    
    # Aumenta o contraste usando CLAHE
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l,a,b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # Reduz ruído
    denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
    
    # Aumenta nitidez
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    return sharpened

# ==========================
# CRIAR HEATMAP DE CONFIANÇA
# ==========================
def criar_heatmap(frame, deteccoes_com_confianca):
    """
    Cria um heatmap mostrando as regiões de maior confiança
    """
    height, width = frame.shape[:2]
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    # Para cada detecção, adiciona intensidade baseada na confiança
    for (x1, y1, x2, y2, conf) in deteccoes_com_confianca:
        # Cria uma máscara gaussiana centrada na detecção
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Tamanho da região de influência
        sigma_x = (x2 - x1) // 2
        sigma_y = (y2 - y1) // 2
        
        # Cria grade de coordenadas
        y_grid, x_grid = np.ogrid[0:height, 0:width]
        
        # Calcula distância gaussiana do centro
        mask = np.exp(-((x_grid - center_x)**2 / (2 * sigma_x**2) + 
                       (y_grid - center_y)**2 / (2 * sigma_y**2)))
        
        # Adiciona ao heatmap ponderado pela confiança
        heatmap += mask * conf
    
    # Normaliza o heatmap
    if heatmap.max() > 0:
        heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
    else:
        heatmap = heatmap.astype(np.uint8)
    
    # Aplica colormap (TURBO ou JET para melhor visualização)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)
    
    # Combina com a imagem original (overlay)
    overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)
    
    return overlay, heatmap_color

# ==========================
# DETECÇÃO MULTI-ESCALA
# ==========================
def detectar_multi_escala(model, frame):
    """
    Tenta detectar placas em diferentes escalas
    Retorna detecções com coordenadas e confiança
    """
    escalas = [1.0, 1.5, 2.0, 0.75, 0.5]
    todas_deteccoes = []
    melhor_confianca = {}
    
    print("[INFO] Testando múltiplas escalas...")
    
    for escala in escalas:
        if escala != 1.0:
            largura = int(frame.shape[1] * escala)
            altura = int(frame.shape[0] * escala)
            frame_redim = cv2.resize(frame, (largura, altura), interpolation=cv2.INTER_CUBIC)
        else:
            frame_redim = frame
        
        # Detecção com threshold reduzido
        results = model(frame_redim, verbose=False, conf=0.2)
        
        if len(results[0].boxes) > 0:
            print(f"  → Escala {escala:.2f}: {len(results[0].boxes)} detecção(ões)")
            
            for box, conf in zip(results[0].boxes.xyxy, results[0].boxes.conf):
                # Ajusta coordenadas de volta para escala original
                x1 = int(box[0] / escala)
                y1 = int(box[1] / escala)
                x2 = int(box[2] / escala)
                y2 = int(box[3] / escala)
                
                # Cria chave única para a região (com tolerância)
                key = (x1//20, y1//20, x2//20, y2//20)
                
                # Mantém apenas a detecção com maior confiança por região
                if key not in melhor_confianca or conf > melhor_confianca[key][4]:
                    melhor_confianca[key] = (x1, y1, x2, y2, float(conf))
    
    # Converte para lista com confiança
    deteccoes_com_conf = list(melhor_confianca.values())
    
    print(f"[INFO] Total de regiões únicas detectadas: {len(deteccoes_com_conf)}")
    return deteccoes_com_conf

# ==========================
# FUNÇÃO PARA OCR LOCAL MELHORADO COM VOTAÇÃO
# ==========================
def ocr_placa(image, debug_index=0):
    """
    Realiza OCR na imagem da placa usando EasyOCR
    Retorna apenas letras e números (formato de placa brasileira)
    Usa sistema de votação para escolher o melhor resultado
    """
    try:
        # Converte para escala de cinza
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Aumenta resolução se muito pequena
        height, width = gray.shape
        if height < 150:
            scale = 150 / height
            new_width = int(width * scale)
            gray = cv2.resize(gray, (new_width, 150), interpolation=cv2.INTER_CUBIC)
        
        # Aumenta contraste
        gray = cv2.equalizeHist(gray)
        
        # Aplica threshold adaptativo
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        
        # Remove ruído com morfologia
        kernel = np.ones((2,2), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
        
        # Inverte cores (tenta fundo escuro, texto claro)
        thresh_inv = cv2.bitwise_not(thresh)
        morph_inv = cv2.bitwise_not(morph)
        
        # Salva versões processadas se DEBUG_MODE
        if DEBUG_MODE:
            cv2.imwrite(f"debug_ocr_{debug_index}_1_gray.jpg", gray)
            cv2.imwrite(f"debug_ocr_{debug_index}_2_thresh.jpg", thresh)
            cv2.imwrite(f"debug_ocr_{debug_index}_3_morph.jpg", morph)
            cv2.imwrite(f"debug_ocr_{debug_index}_4_thresh_inv.jpg", thresh_inv)
            cv2.imwrite(f"debug_ocr_{debug_index}_5_morph_inv.jpg", morph_inv)
        
        # SISTEMA DE VOTAÇÃO: Tenta OCR em TODAS as versões
        versoes = {
            'gray': gray,
            'thresh': thresh,
            'morph': morph,
            'thresh_inv': thresh_inv,
            'morph_inv': morph_inv
        }
        
        votos = {}  # Dicionário para contar votos de cada texto
        detalhes_votos = []  # Lista para debug
        
        print(f"    [VOTAÇÃO] Testando {len(versoes)} versões da imagem...")
        
        for nome_versao, imagem_versao in versoes.items():
            resultados = reader.readtext(imagem_versao, detail=0, paragraph=False, 
                                        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            
            for texto in resultados:
                # Remove espaços e caracteres especiais
                texto_limpo = re.sub(r'[^A-Z0-9]', '', texto.upper())
                
                if len(texto_limpo) >= 6:  # Placa tem no mínimo 6 caracteres
                    # Limita a 7 caracteres
                    if len(texto_limpo) > 7:
                        texto_limpo = texto_limpo[:7]
                    
                    # Adiciona voto
                    if texto_limpo not in votos:
                        votos[texto_limpo] = 0
                    votos[texto_limpo] += 1
                    
                    detalhes_votos.append((nome_versao, texto_limpo))
        
        if not votos:
            print(f"    [VOTAÇÃO] ❌ Nenhum texto válido encontrado")
            return "SEM TEXTO"
        
        # Ordena candidatos por número de votos
        candidatos_ordenados = sorted(votos.items(), key=lambda x: x[1], reverse=True)
        
        # Mostra resultado da votação
        print(f"    [VOTAÇÃO] Resultados:")
        for candidato, num_votos in candidatos_ordenados[:3]:  # Top 3
            percentual = (num_votos / len(detalhes_votos)) * 100
            print(f"      • {candidato}: {num_votos} votos ({percentual:.1f}%)")
        
        # Escolhe o vencedor
        vencedor, votos_vencedor = candidatos_ordenados[0]
        
        # Se houver empate, usa o mais longo
        empate = [c for c, v in candidatos_ordenados if v == votos_vencedor]
        if len(empate) > 1:
            vencedor = max(empate, key=len)
            print(f"    [VOTAÇÃO] ⚖️  Empate! Escolhendo mais longo: {vencedor}")
        else:
            print(f"    [VOTAÇÃO] 🏆 Vencedor: {vencedor}")
        
        return vencedor
        
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
    
    # Pré-processa a imagem
    frame_processado = pre_processar_imagem(frame)
    
    if DEBUG_MODE:
        cv2.imwrite("debug_preprocessed.jpg", frame_processado)
        print("[DEBUG] Imagem pré-processada salva: debug_preprocessed.jpg")
    
    # Detecção multi-escala (agora retorna com confiança)
    deteccoes_com_conf = detectar_multi_escala(model, frame_processado)
    
    if len(deteccoes_com_conf) == 0:
        print("[AVISO] Nenhuma placa detectada!")
        
        # Tenta com a imagem original (sem pré-processamento)
        print("[INFO] Tentando com imagem original...")
        results = model(frame, verbose=False, conf=0.2)
        
        if len(results[0].boxes) > 0:
            print(f"[INFO] {len(results[0].boxes)} placa(s) detectada(s) na imagem original")
            deteccoes_com_conf = [(int(box[0]), int(box[1]), int(box[2]), int(box[3]), float(conf)) 
                                  for box, conf in zip(results[0].boxes.xyxy, results[0].boxes.conf)]
        else:
            output_path = "resultado_sem_placa.jpg"
            cv2.imwrite(output_path, frame)
            print(f"[INFO] Nenhuma detecção encontrada. Imagem salva em: {output_path}")
            return
    
    # CRIA HEATMAP
    if ENABLE_HEATMAP:
        print("[INFO] Gerando heatmap de confiança...")
        heatmap_overlay, heatmap_puro = criar_heatmap(frame.copy(), deteccoes_com_conf)
        cv2.imwrite("heatmap_overlay.jpg", heatmap_overlay)
        cv2.imwrite("heatmap_puro.jpg", heatmap_puro)
        print("[✅] Heatmap salvo: heatmap_overlay.jpg e heatmap_puro.jpg")
    
    print(f"\n[INFO] Processando {len(deteccoes_com_conf)} detecção(ões)...")
    
    # Lista para armazenar os resultados
    placas_detectadas = []
    
    # Percorre as detecções encontradas
    for i, (x1, y1, x2, y2, conf) in enumerate(deteccoes_com_conf):
        print(f"\n[INFO] Processando detecção {i+1}/{len(deteccoes_com_conf)} (confiança: {conf:.2%})...")
        
        # Adiciona margem ao recorte para melhorar OCR
        margin = 10
        x1_crop = max(0, x1 - margin)
        y1_crop = max(0, y1 - margin)
        x2_crop = min(frame.shape[1], x2 + margin)
        y2_crop = min(frame.shape[0], y2 + margin)
        
        placa_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
        
        if placa_crop.size == 0:
            print(f"[AVISO] Recorte {i} está vazio, pulando...")
            continue
        
        # Salva o recorte
        crop_path = f"placa_{i}.jpg"
        cv2.imwrite(crop_path, placa_crop)
        print(f"  → Recorte salvo: {crop_path}")
        
        # OCR LOCAL MELHORADO COM VOTAÇÃO
        print(f"  → Lendo texto da placa...")
        texto = ocr_placa(placa_crop, debug_index=i)
        print(f"  ✅ Resultado: {texto}")
        
        # Armazena o resultado
        placas_detectadas.append(texto)
        
        # Desenha caixa verde e texto na imagem original
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Adiciona percentual de confiança no canto da caixa
        conf_text = f"{conf:.1%}"
        cv2.putText(frame, conf_text, (x2 - 60, y2 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
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
    if ENABLE_HEATMAP:
        print(f"[🔥] Heatmap salvo em: heatmap_overlay.jpg e heatmap_puro.jpg")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
