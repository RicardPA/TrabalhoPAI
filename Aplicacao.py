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

clicked = False
root = Tk.Tk()
root.title("TrabalhoPAI");

def abrir_imagem():
    global img, imgCopy

    fechar_menu()

    fileName = filedialog.askopenfilename(filetypes= (("PNG","*.png"), ("JPG","*.jpg")))
    img = cv2.imread(fileName)
    cv2.namedWindow("Imagem")
    cv2.imshow("Imagem", img)
    cv2.setMouseCallback("Imagem", cortar_imagem)

def comparar():
    fechar_menu()

"""Metodo para fazer corte de imagem com o mouse"""
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
            construir_menu_salvar_imagem()

# --- --- --- --- --- --- --- --- ---

"Funcoes para controle de apresentacao da tela"
def fechar_menu():
    root.withdraw()

def abrir_menu():
    root.deiconify()

def limpar_menu():
    for widget in root.winfo_children():
        widget.destroy()

def construir_menu_principal():
    screen = Tk.Canvas(root, height = 500, width= 500, bg = "#202020")
    screen.pack()

    btn_abrir_arquivo = Tk.Button(root, text = "Cortar Imagem", padx = 2, pady = 2, fg = "white", bg = "#00006F", command = abrir_imagem)
    btn_abrir_arquivo.place(relwidth = 0.2, relheight = 0.05, relx = 0.02, rely = 0.02)

    btn_comparar = Tk.Button(root, text = "Comparar", padx = 2, pady = 2, fg = "white", bg = "#00006F", command = comparar)
    btn_comparar.place(relwidth = 0.2, relheight = 0.05, relx = 0.23, rely = 0.02)

def construir_menu_salvar_imagem():
    screen = Tk.Canvas(root, height = 500, width= 500, bg = "#202020")
    screen.pack()

    btn_abrir_arquivo = Tk.Button(root, text = "Cortar Imagem", padx = 2, pady = 2, fg = "white", bg = "#00006F", command = abrir_imagem)
    btn_abrir_arquivo.place(relwidth = 0.2, relheight = 0.05, relx = 0.02, rely = 0.02)

    btn_comparar = Tk.Button(root, text = "Comparar", padx = 2, pady = 2, fg = "white", bg = "#00006F", command = comparar)
    btn_comparar.place(relwidth = 0.2, relheight = 0.05, relx = 0.23, rely = 0.02)

construir_menu_principal()
root.mainloop()