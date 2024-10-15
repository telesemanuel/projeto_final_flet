import flet as ft
import cv2
import numpy as np
import os

largura = 220
altura = 220

def get_imagem_com_id():
    caminhos = [os.path.join('fotos', f) for f in os.listdir('fotos')]
    faces = []
    ids = []
    
    for caminho_imagem in caminhos:
        imagem_face = cv2.cvtColor(cv2.imread(caminho_imagem), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(caminho_imagem)[-1].split('.')[1])
        ids.append(id)
        faces.append(imagem_face)

    return np.array(ids), faces

def captura(largura, altura, page, id):
    classificador = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    classificador_olho = cv2.CascadeClassifier('haarcascade_eye.xml')

    if not os.path.exists('fotos'):
        os.makedirs('fotos')

    camera = cv2.VideoCapture(0)
    amostra = 1
    n_amostras = 25

    page.controls.append(ft.Text(f"Capturando as imagens para ID {id}..."))
    page.update()

    try:
        while True:
            conectado, imagem = camera.read()
            if not conectado:
                raise ValueError("Não foi possível capturar a imagem.")
            
            imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
            faces_detectadas = classificador.detectMultiScale(imagem_cinza, scaleFactor=1.5, minSize=(150, 150))

            for (x, y, l, a) in faces_detectadas:
                cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
                regiao = imagem[y:y + a, x:x + l]
                regiao_cinza_olho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
                olhos_detectados = classificador_olho.detectMultiScale(regiao_cinza_olho)

                for (ox, oy, ol, oa) in olhos_detectados:
                    cv2.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (0, 255, 0), 2)

                if np.average(imagem_cinza) > 110 and amostra <= n_amostras:
                    imagem_face = cv2.resize(imagem_cinza[y:y + a, x:x + l], (largura, altura))
                    caminho = f'fotos/pessoa.{id}.{amostra}.jpg'
                    cv2.imwrite(caminho, imagem_face)
                    page.controls.append(ft.Text(f'[foto] {amostra} de {id} capturada com sucesso em {caminho}.'))
                    amostra += 1
                    page.update()

            cv2.imshow('Detectar faces', imagem)
            cv2.waitKey(1)

            if amostra > n_amostras:
                page.controls.append(ft.Text("Faces capturadas com sucesso."))
                page.update()
                break
            elif cv2.waitKey(1) & 0xFF == ord('q'):
                page.controls.append(ft.Text('Câmera encerrada.'))
                page.update()
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

def treinamento(page):
    eigenface = cv2.face.EigenFaceRecognizer_create()
    fisherface = cv2.face.FisherFaceRecognizer_create()
    lbph = cv2.face.LBPHFaceRecognizer_create()

    ids, faces = get_imagem_com_id()

    page.controls.append(ft.Text('Treinando...'))
    page.update()

    eigenface.train(faces, ids)
    eigenface.write('classificadorEigen.yml')
    fisherface.train(faces, ids)
    fisherface.write('classificadorFisher.yml')
    lbph.train(faces, ids)
    lbph.write('classificadorLBPH.yml')

    page.controls.append(ft.Text('Treinamento finalizado com sucesso!'))
    page.update()

def reconhecedor_eigenfaces(largura, altura, page):
    detector_faces = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    reconhecedor = cv2.face.EigenFaceRecognizer_create()
    reconhecedor.read("classificadorEigen.yml")
    fonte = cv2.FONT_HERSHEY_COMPLEX_SMALL

    camera = cv2.VideoCapture(0)

    try:
        while True:
            conectado, imagem = camera.read()
            if not conectado:
                raise ValueError("Não foi possível capturar a imagem.")
            
            imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
            faces_detectadas = detector_faces.detectMultiScale(imagem_cinza, scaleFactor=1.5, minSize=(30, 30))

            for (x, y, l, a) in faces_detectadas:
                imagem_face = cv2.resize(imagem_cinza[y:y + a, x:x + l], (largura, altura))
                cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
                id, confianca = reconhecedor.predict(imagem_face)
                cv2.putText(imagem, str(id), (x, y + (a + 30)), fonte, 2, (0, 0, 255))

            cv2.imshow("Reconhecer faces", imagem)
            if cv2.waitKey(1) == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

def main(page: ft.Page):
    page.title = "Sistema de Reconhecimento Facial"

    id_input = ft.TextField(label="Digite o ID do usuário")
    page.add(id_input)

    btn_captura = ft.ElevatedButton(
        text="Capturar Imagem",
        on_click=lambda e: captura(largura, altura, page, id_input.value),
    )

    btn_treino = ft.ElevatedButton(
        text="Treinar Sistema",
        on_click=lambda e: treinamento(page),
    )

    btn_reconhecimento = ft.ElevatedButton(
        text="Reconhecer Face",
        on_click=lambda e: reconhecedor_eigenfaces(largura, altura, page),
    )

    page.add(btn_captura, btn_treino, btn_reconhecimento)

ft.app(target=main)
