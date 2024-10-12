import flet as ft
import cv2
import os

# Funções do sistema de reconhecimento facial (substituir pelas funções atuais)
def captura_imagem(id_usuario, largura, altura):
    # Aqui você chamaria a função captura(largura, altura)
    print(f"Capturando imagens para o ID: {id_usuario}")

def treinar_sistema():
    # Aqui você chamaria a função treinamento()
    print("Treinando o sistema...")

def reconhecer_face(largura, altura):
    # Aqui você chamaria a função reconhecedor_eigenfaces(largura, altura)
    print("Reconhecendo faces...")

# Função principal da interface Flet
def main(page: ft.Page):
    page.title = "Reconhecimento Facial - Sistema"
    
    # Campo para digitar o ID do usuário
    id_usuario = ft.TextField(label="Digite o ID do usuário")
    
    # Botão para capturar imagem
    btn_capturar = ft.ElevatedButton(text="Capturar Imagem", on_click=lambda _: captura_imagem(id_usuario.value, 220, 220))
    
    # Botão para treinar o sistema
    btn_treinar = ft.ElevatedButton(text="Treinar Sistema", on_click=lambda _: treinar_sistema())
    
    # Botão para reconhecer face
    btn_reconhecer = ft.ElevatedButton(text="Reconhecer Face", on_click=lambda _: reconhecer_face(220, 220))
    
    # Adicionando os componentes à página
    page.add(
        ft.Column(
            controls=[
                id_usuario,
                btn_capturar,
                btn_treinar,
                btn_reconhecer
            ]
        )
    )

# Rodar o app
ft.app(target=main)
