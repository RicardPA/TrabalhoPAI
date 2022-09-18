"""
Ciencias da Computação/Praça da Liberdade

Alunos: 
* Lucas Satlher Campos Lacerda 
* Ricardo Portilho de Andrade
"""

from asyncio.windows_events import NULL
from logging import root
import tkinter as Tk
from tkinter import filedialog
import cv2
import numpy as np

clicked = False # Verificar comeco de corte
root = Tk.Tk() # Iniciar tela de menu
root.title("TrabalhoPAI"); # Alterar nome da tela
root.minsize(500, 50)
root.maxsize(500, 50)

"""
Nome: abrir_imagem
Funcao: Abrir uma imagem e permitir que o usuario
corte uma regiao e salve a porte cortada. 
"""
def abrir_imagem():
    global img, imgCopy

    fechar_menu()
    fileName = filedialog.askopenfilename(filetypes= (("PNG","*.png"), ("JPG","*.jpg")))
    img = cv2.imread(fileName)

    if img == None:
        abrir_menu()
    else:
        cv2.namedWindow("Imagem")
        cv2.imshow("Imagem", img)
        cv2.setMouseCallback("Imagem", cortar_imagem)
    

"""
Nome: comparar
Funcao: Comparar uma imagem com partes de outra 
selecionando e apresentando a parte mais parecida. 
"""
def comparar():
    fechar_menu()

"""
Nome: cortar_imagem
Funcao: Identificar a regiao de corte e retornar
a imagem da regiao.
"""
def cortar_imagem(mouse, x, y, flags, param):
    global clicked, start_x, start_y, end_x, end_y, img, imgCopy, imgCut

    if mouse == cv2.EVENT_LBUTTONDOWN:
        start_x, start_y, end_x, end_y = x, y, x, y
        clicked = True

    elif mouse == cv2.EVENT_MOUSEMOVE:
        if clicked == True:
            end_x, end_y = x, y
            imgCopy = img.copy() 
            cv2.rectangle(imgCopy, (start_x, start_y),(end_x, end_y), (255, 0, 0), 2)
            cv2.imshow("Imagem", imgCopy)

    elif mouse == cv2.EVENT_LBUTTONUP:
        end_x, end_y = x, y
        clicked = False 
        points = [(start_x, start_y),(end_x, end_y)]

        if len(points) == 2: 
            imgCut = img[points[0][1]:points[1][1], points[0][0]:points[1][0]]
            cv2.imwrite('processada.png',imgCut)
            cv2.destroyAllWindows()
            cv2.namedWindow("Cortada")
            cv2.imshow("Cortada", imgCut)
            if imgCut != None:
                construir_menu_salvar_imagem()
            else:
                construir_menu_principal()

"""
Nome: salvar_imagem_gerada
Funcao: Salvar imagem gerada pelo corte.
"""
def salvar_imagem_gerada():
    print("SALVAR")

"""
Nome: nao_salvar_imagem_gerada
Funcao: Nao salvar imagem rerada pelo corte.
"""
def nao_salvar_imagem_gerada():
    print("NAO SALVAR")


# --- --- --- --- TELA PRINCIPAL / MENU --- --- --- ---

"""
Nome: fechar_menu
Funcao: Fechar menu principal.
"""
def fechar_menu():
    root.withdraw()

"""
Nome: abrir_menu
Funcao: Abrir menu principal.
"""
def abrir_menu():
    root.deiconify()

"""
Nome: limpar_menu
Funcao: Resetar configuracoes do menu.
"""
def limpar_menu():
    for widget in root.winfo_children():
        widget.destroy()

"""
Nome: construir_menu_principal
Funcao: Criar menu com configuração principal.
"""
def construir_menu_principal():
    # Configurar tela
    screen = Tk.Canvas(root, height = 50, width= 500, bg = "#202020")
    screen.pack()

    # Criar botao responsavel por chamar o corte de uma imagem
    btn_abrir_arquivo = Tk.Button(root, text = "Cortar Imagem", padx = 1, pady = 1, fg = "white", bg = "#00006F", command = abrir_imagem)
    btn_abrir_arquivo.place(relwidth = 0.2, relheight = 0.8, relx = 0.02, rely = 0.1)

    # Criar botao resonsavel por selecionar imagens que serao comparadas 
    btn_comparar = Tk.Button(root, text = "Comparar", padx = 1, pady = 1, fg = "white", bg = "#00006F", command = comparar)
    btn_comparar.place(relwidth = 0.2, relheight = 0.8, relx = 0.23, rely = 0.1)

"""
Nome: construir_menu_principal
Funcao: Criar menu com configuração reponsavel
por salvar a imagem obtida pelo corte.
"""
def construir_menu_salvar_imagem():
    screen = Tk.Canvas(root, height = 50, width= 500, bg = "#202020")
    screen.pack()

    btn_salvar = Tk.Button(root, text = "Salvar Imagem", padx = 2, pady = 2, fg = "white", bg = "GREEN", command = salvar_imagem_gerada)
    btn_salvar.place(relwidth = 0.2, relheight = 0.05, relx = 0.02, rely = 0.02)

    btn_nao_salvar = Tk.Button(root, text = "Não Salvar", padx = 2, pady = 2, fg = "white", bg = "RED", command = nao_salvar_imagem_gerada)
    btn_nao_salvar.place(relwidth = 0.2, relheight = 0.05, relx = 0.23, rely = 0.02)

construir_menu_principal() # Chamar primeira criacao/configuracao de tela
root.mainloop() # Deixar tela aberta