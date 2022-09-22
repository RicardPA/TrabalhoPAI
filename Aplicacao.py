"""
Ciencias da Computação/Praça da Liberdade

Alunos: 
* Lucas Satlher Campos Lacerda 
* Ricardo Portilho de Andrade
"""

from PIL import Image, ImageFilter 
from asyncio.windows_events import NULL
from logging import root
import tkinter as Tk
from tkinter import filedialog
import cv2
import os
import numpy as np

clicked = False # Verificar comeco de corte
root = Tk.Tk() # Iniciar tela de menu

"""
Nome: abrir_imagem
Funcao: Abrir uma imagem e permitir que o usuario
corte uma regiao e salve a porte cortada. 
"""
def abrir_imagem():
    global img, imgCopy, extensionFile

    fechar_menu()

    # Obter arquivo selecionado pelo usuario
    fileName = filedialog.askopenfilename(filetypes= (("PNG","*.png"), ("JPG","*.jpg")))
    img = cv2.imread(fileName)

    # Obter extensao do arquivo selecionado
    if (fileName[-4] == "."):
        extensionFile = "." + fileName.split(".")[1] 

    # Rcomecar o menu caso nao exista imagem
    if img is None:
        abrir_menu()

    # Disponibilizar corte da imagem 
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
    global firstImg, secondImg, saveImage, extensionFile
    extensionFiletypes = [('PNG', "*.png"), ('JPG', "*.jpg")]

    fechar_menu()

    # Obter primeiro arquivo selecionado pelo usuario
    fileNameFirst = filedialog.askopenfilename(title = "Selecione a primeira imagem para comparação", filetypes= extensionFiletypes)
    firstImg = cv2.imread(fileNameFirst)

    if firstImg is None:
        abrir_menu()
    else:
        if (fileNameFirst[-4] == "."):
            typeFileName = fileNameFirst.split(".")[1].upper() 
            extensionFile = "." + fileNameFirst.split(".")[1]
            pathAndNameFirstImage = fileNameFirst.split(".")[0]

            # Reconhecer a extensao do arquivo e limitar proxima escolha
            if(typeFileName == "PNG"):
                extensionFiletypes = [('PNG', "*.png")]
            else:
                extensionFiletypes = [('JPG', "*.jpg")]

        # Obter segundo arquivo selecionado pelo usuario
        fileNameSecond = filedialog.askopenfilename(title = "Selecione a segunda imagem para comparação", filetypes= extensionFiletypes)
        secondImg = cv2.imread(fileNameSecond)

        if secondImg is None:
            abrir_menu()
        else:
            if (fileNameSecond[-4] == "."):
                pathAndNameSecondImage = fileNameSecond.split(".")[0]

            imageProcessFirst = Image.open(r""+fileNameFirst)
            imageProcessSecond = Image.open(r""+fileNameSecond) 

            imageProcessFirst = imageProcessFirst.convert("L") 
            imageProcessFirst = imageProcessFirst.filter(ImageFilter.FIND_EDGES) 
            imageProcessFirst.save(r""+pathAndNameFirstImage+"Process"+extensionFile)

            imageProcessSecond = imageProcessSecond.convert("L") 
            imageProcessSecond = imageProcessSecond.filter(ImageFilter.FIND_EDGES) 
            imageProcessSecond.save(r""+pathAndNameSecondImage+"Process"+extensionFile)

            processFirstImg = cv2.imread(pathAndNameFirstImage+"Process"+extensionFile)
            processSecondImg = cv2.imread(pathAndNameSecondImage+"Process"+extensionFile)

            os.remove(pathAndNameFirstImage+"Process"+extensionFile)
            os.remove(pathAndNameSecondImage+"Process"+extensionFile)

            processFirstImg = processFirstImg - 13
            processSecondImg = processSecondImg - 13

            resultComparation = cv2.matchTemplate(processSecondImg, processFirstImg, cv2.TM_CCOEFF_NORMED)
            (valueMin, valueMax, comparationMin, comparationMax) = cv2.minMaxLoc(resultComparation)

            cv2.imshow("IMG(1)", processFirstImg)
            cv2.imshow("IMG(2)", processSecondImg)

            (valor_inicial_coordenada_x, valor_inicial_coordenada_y) = comparationMax
            valor_final_coordenada_x = valor_inicial_coordenada_x + firstImg.shape[1]
            valor_final_coordenada_y = valor_inicial_coordenada_y + firstImg.shape[0]
            
            resultImage = secondImg.copy()
            cv2.rectangle(resultImage, (valor_inicial_coordenada_x, valor_inicial_coordenada_y), (valor_final_coordenada_x, valor_final_coordenada_y), (255, 0, 0), 2)
            cv2.imshow("Resultado da comparação", resultImage)
            saveImage = resultImage.copy()

            if resultImage is None:
                abrir_menu()
                construir_menu_principal()
            else:    
                abrir_menu()
                construir_menu_salvar_imagem()

"""
Nome: cortar_imagem
Funcao: Identificar a regiao de corte e retornar
a imagem da regiao.
"""
def cortar_imagem(mouse, position_x, position_y, flags, param):
    global clicked, start_x, start_y, end_x, end_y, img, imgCopy, imgCut, saveImage

    # Reconhecer primeiro clique do usuario
    if mouse == cv2.EVENT_LBUTTONDOWN:
        # Obter posicoes iniciais e iniciar verificacao de movimento
        start_x, start_y, end_x, end_y = position_x, position_y, position_x, position_y
        clicked = True

    # Reconhecer movimento do mouse apos o clique 
    elif mouse == cv2.EVENT_MOUSEMOVE:
        # Verificar se primeiro clique foi feito
        if clicked == True:
            end_x, end_y = position_x, position_y # Obter posicoes atuais
            # Apresentar uma copia da imagem com retangulo representando o corte
            imgCopy = img.copy() 
            cv2.rectangle(imgCopy, (start_x, start_y),(end_x, end_y), (255, 0, 0), 2)
            cv2.imshow("Imagem", imgCopy)

    # Reconhecer a ultima posicao do corte
    elif mouse == cv2.EVENT_LBUTTONUP:
        end_x, end_y = position_x, position_y # Pegar ultima posicao
        clicked = False # Mudar valor para futuros cortes
        points = [(start_x, start_y),(end_x, end_y)] # Pontos do corte

        if len(points) == 2: 
            # Cortar imagem e apresentar imagem cortada
            imgCut = img[points[0][1]:points[1][1], points[0][0]:points[1][0]]
            cv2.destroyAllWindows()
            cv2.namedWindow("Imagem Cortada")
            cv2.imshow("Imagem Cortada", imgCut)
            saveImage = imgCut.copy()

            # Caso nenhuma imagem seja apresentada volta para o menu principal
            if imgCut is None:
                abrir_menu()
                construir_menu_principal()
            
            # Caso a imagem seja apresentada abre a notificacao de salvar 
            else:
                abrir_menu()
                construir_menu_salvar_imagem()

"""
Nome: salvar_imagem_gerada
Funcao: Salvar imagem gerada pelo corte.
"""
def salvar_imagem_gerada():
    typeFileName = extensionFile.split(".")[1].upper()
    extensionFiletypes = []
    indexFilePath = 0
    nameImg = ""
    
    # Reconhecer a extensao do arquivo
    if(typeFileName == "PNG"):
        extensionFiletypes = [('PNG', "*.png")]
    else:
        extensionFiletypes = [('JPG', "*.jpg")]

    # Obter nome do arquivo e posicao do arquivo
    filepathAndFile = filedialog.asksaveasfilename(title = "Selecione a pasta para armazenar o arquivo", filetypes = extensionFiletypes)
    
    for char in filepathAndFile[::-1]:
        if(char != "/"):
            indexFilePath = indexFilePath + 1
        else:
            break
    
    nameImg = filepathAndFile[-indexFilePath:]

    if (nameImg[-4] == "."):
        nameImg = nameImg.split(".")[0]
    
    # Obter caminho do arquivo
    filepath = filepathAndFile[0:-indexFilePath]
    
    # Salvar arquivo
    os.chdir(filepath)
    cv2.imwrite(nameImg + extensionFile, saveImage)
    cv2.destroyAllWindows()
    construir_menu_principal()

"""
Nome: nao_salvar_imagem_gerada
Funcao: Nao salvar imagem rerada pelo corte.
"""
def nao_salvar_imagem_gerada():
    cv2.destroyAllWindows()
    construir_menu_principal()
    
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
    limpar_menu()
    root.title("Menu (Trabalho de PAI)");
    root.minsize(500, 50)
    root.maxsize(500, 50)
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
por salvar a imagem obtida pelo corte e a imagem 
resultante da comparacao.
"""
def construir_menu_salvar_imagem():
    # Configurar tela
    limpar_menu()
    root.title("Aviso");
    root.minsize(220, 50)
    root.maxsize(220, 50)
    screen = Tk.Canvas(root, height = 50, width= 220, bg = "#202020")
    screen.pack()

    # Criar botao com opcao para salvar
    btn_salvar = Tk.Button(root, text = "Salvar Imagem", padx = 1, pady = 1, fg = "white", bg = "GREEN", command = salvar_imagem_gerada)
    btn_salvar.place(relwidth = 0.4, relheight = 0.8, relx = 0.02, rely = 0.1)

    # Criar botao com opcao de nao salvar 
    btn_nao_salvar = Tk.Button(root, text = "Não Salvar", padx = 1, pady = 1, fg = "white", bg = "RED", command = nao_salvar_imagem_gerada)
    btn_nao_salvar.place(relwidth = 0.4, relheight = 0.8, relx = 0.585, rely = 0.1)

construir_menu_principal() # Chamar primeira criacao/configuracao de tela
root.mainloop() # Deixar tela aberta