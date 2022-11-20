"""
Ciencias da Computação/Praça da Liberdade

Alunos: 
* Lucas Satlher Campos Lacerda 
* Ricardo Portilho de Andrade
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torchvision import models
from torchsummary import summary
from torch import nn, optim
import torch.nn.functional as functions
import torchvision

import numpy as np
import pandas as pd

import datetime as dt
import math
import os
import sys
import tarfile
import time
import tkinter as Tk
import warnings
from asyncio.windows_events import NULL
from logging import root
from tkinter import filedialog
import cv2
import matplotlib
import matplotlib.pyplot as plt
import tqdm
from PIL import Image, ImageFilter
from matplotlib.pyplot import figure

warnings.filterwarnings("ignore")

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
    
    if fileName != "":
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
    else:
        abrir_menu()
    

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

    # Rcomecar o menu caso nao exista imagem
    if firstImg is None:
        abrir_menu()
    
    # Abrir segunda imagem e fazer comparacao
    else:
        firstImg = cv2.cvtColor(firstImg, cv2.COLOR_BGR2GRAY)
        firstImg = cv2.equalizeHist(firstImg)

        # Obter extensao para obter proxima imgem
        if (fileNameFirst[-4] == "."):
            typeFileName = fileNameFirst.split(".")[1].upper() 
            extensionFile = "." + fileNameFirst.split(".")[1]

            # Reconhecer a extensao do arquivo e limitar proxima escolha
            if(typeFileName == "PNG"):
                extensionFiletypes = [('PNG', "*.png")]
            else:
                extensionFiletypes = [('JPG', "*.jpg")]

        # Obter segundo arquivo selecionado pelo usuario
        fileNameSecond = filedialog.askopenfilename(title = "Selecione a segunda imagem para comparação", filetypes= extensionFiletypes)
        secondImg = cv2.imread(fileNameSecond)

        # Rcomecar o menu caso nao exista imagem
        if secondImg is None:
            abrir_menu()

        # Fazer comparacao
        else:
            secondImg = cv2.cvtColor(secondImg, cv2.COLOR_BGR2GRAY)
            secondImg = cv2.equalizeHist(secondImg)

            secondImg = secondImg[
                int(secondImg.shape[0]*0.20):int(secondImg.shape[0]*0.75), 
                int(secondImg.shape[1]*0.05):int(secondImg.shape[1]*0.95)
            ]

            # Comparar imagens
            resultComparation = cv2.matchTemplate(secondImg, firstImg, cv2.TM_CCORR_NORMED)
            (valueMin, valueMax, comparationMin, comparationMax) = cv2.minMaxLoc(resultComparation)

            # Obter coordenadas do local de semelhanca da imagem
            (start_x, start_y) = comparationMax
            end_x = start_x + firstImg.shape[1]
            end_y = start_y + firstImg.shape[0]
            
            # Apresentar imagem com comparacao
            resultImage = cv2.imread(fileNameSecond)
            cv2.rectangle(resultImage, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
            cv2.imshow("Resultado da comparação", resultImage)
            saveImage = resultImage.copy()

            # Reiniciar menu
            if resultImage is None:
                abrir_menu()
                construir_menu_principal()
            
            # Permitir salvar imagem com comparacao
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

"""
Nome: aumentar_dados
Funcao: Aumentar os dados utilizados na aplicacao
"""
def aumentar_dados():
    filepath = filedialog.askdirectory()
    filepathDestinyMirrored = filepath + "Mirrored" 
    filepathDestinyEqualizeHist = filepath + "EqualizeHist" 

    for _, _, arquivos in os.walk(filepath):
        os.chdir(filepath)

        for arquivo in arquivos:  
            imagemEqualizeHist = cv2.imread(arquivo)
            imagemEqualizeHist = cv2.cvtColor(imagemEqualizeHist, cv2.COLOR_BGR2GRAY)
            imagemEqualizeHist = cv2.equalizeHist(imagemEqualizeHist)
            imageEspelhada = Image.open(r""+arquivo)
            imageEspelhada = imageEspelhada.transpose(Image.FLIP_LEFT_RIGHT)

            if os.path.exists(filepathDestinyEqualizeHist):
                os.chdir(filepathDestinyEqualizeHist)
                cv2.imwrite("h"+arquivo, imagemEqualizeHist)
                os.chdir(filepath)
            else:
                os.mkdir(filepathDestinyEqualizeHist)
                os.chdir(filepathDestinyEqualizeHist)
                cv2.imwrite("h"+arquivo, imagemEqualizeHist)
                os.chdir(filepath)

            if arquivo.find('R') >= 0:
                arquivo = arquivo.replace("R", "L")
            elif arquivo.find('L') >= 0:
                arquivo = arquivo.replace("L", "R")

            if os.path.exists(filepathDestinyMirrored):
                os.chdir(filepathDestinyMirrored)
                imageEspelhada.save(r""+"c"+arquivo)
                os.chdir(filepath)
            else:
                os.mkdir(filepathDestinyMirrored)
                os.chdir(filepathDestinyMirrored)
                imageEspelhada.save(r""+"c"+arquivo)
                os.chdir(filepath)

"""
Nome: shufflenet
Funcao: Aplicar o aprendizado shufflenet
"""
def shufflenet():
    print("Em construcao")

class ShuffleBlock(nn.Module):
  def __init__(self, groups):
    super(ShuffleBlock, self).__init__()
    self.groups = groups
  def forward(self, x):
    N,C,H,W = x.size()
    g = self.groups
    return x.view(N, g, C//g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)

class Bottleneck(nn.Module): 
  def __init__(self, in_planes, out_planes, stride, groups):
    super(Bottleneck, self).__init__()
    self.stride = stride
    mid_planes = int(out_planes/4)
    g = 1 if in_planes == 24 else groups
    self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, groups=g, bias=False)
    self.bn1 = nn.BatchNorm2d(mid_planes)
    self.shuffle1 = ShuffleBlock(groups= g)
    self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes, bias=False)
    self.bn2 = nn.BatchNorm2d(mid_planes)
    self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, groups=groups, bias=False)
    self.bn3 = nn.BatchNorm2d(out_planes)
    self.shortcut = nn.Sequential()
    if stride==2:
      self.shortcut = nn.Sequential(nn.AvgPool2d(3,stride=2, padding =1))
  def forward(self,x):
    out = functions.relu(self.bn1(self.conv1(x)))
    out = self.shuffle1(out)
    out = functions.relu(self.bn2(self.conv2(out)))
    out = self.bn3(self.conv3(out))
    res = self.shortcut(x)
    out = functions.relu(torch.cat([out,res], 1)) if self.stride==2 else functions.relu(out+res)
    return out

class ShuffleNet(nn.Module):
  def __init__(self, cfg):
    super(ShuffleNet, self).__init__()
    out_planes = cfg['out_planes']
    num_blocks = cfg['num_blocks']
    groups = cfg['groups']
    self.conv1 = nn.Conv2d(3, 24, kernel_size=1, bias = False)
    self.bn1 = nn.BatchNorm2d(24)
    self.in_planes = 24
    self.layer1 = self._make_layer(out_planes[0], num_blocks[0], groups)
    self.layer2 = self._make_layer(out_planes[1], num_blocks[1], groups)
    self.layer3 = self._make_layer(out_planes[2], num_blocks[2], groups)
    self.linear = nn.Linear(out_planes[2], 10) #10 as there are 10 classes

  def _make_layer(self, out_planes, num_blocks, groups):
    layers = []
    for i in range(num_blocks):
      stride = 2 if i == 0 else 1
      cat_planes = self.in_planes if i==0 else 0
      layers.append(Bottleneck(self.in_planes, out_planes-cat_planes, stride=stride, groups=groups))
      self.in_planes = out_planes
    return nn.Sequential(*layers)
  
  def forward(self,x):
    out = functions.relu(self.bn1(self.conv1(x)))
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = functions.avg_pool2d(out, 4)
    out = out.view(out.size(0), -1)
    out = self.linear(out)
    return out

def ShuffleNetG2():
  cfg = {'out_planes': [200, 400, 800],
         'num_blocks': [4, 8, 4],
         'groups': 2
         }
  return ShuffleNet(cfg)

def ShuffleNetG3():
  cfg = {'out_planes': [240, 480, 960],
         'num_blocks': [4, 8, 4],
         'groups': 3
         }
  return ShuffleNet(cfg)

#Shufflenet with groups = 2
net2 = ShuffleNetG2()
print("ShuffleNet with 2 Groups: " + str(net2))

#Shufflenet with groups = 3
net3 = ShuffleNetG3()
print("ShuffleNet with 3 Groups: " + str(net3))
#we will be using g=3 for training

#Setting the model with CUDA
if torch.cuda.is_available():
  net3.cuda()

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
    root.minsize(450, 50)
    root.maxsize(450, 50)
    screen = Tk.Canvas(root, height = 50, width = 450, bg = "#202020")
    screen.pack()

    # Criar botao responsavel por chamar o corte de uma imagem
    btn_abrir_arquivo = Tk.Button(root, text = "Cortar Imagem", padx = 1, pady = 1, fg = "white", bg = "#00006F", command = abrir_imagem)
    btn_abrir_arquivo.place(relwidth = 0.22, relheight = 0.8, relx = 0.02, rely = 0.1)

    # Criar botao resonsavel por selecionar imagens que serao comparadas 
    btn_comparar = Tk.Button(root, text = "Comparar", padx = 1, pady = 1, fg = "white", bg = "#00006F", command = comparar)
    btn_comparar.place(relwidth = 0.20, relheight = 0.8, relx = 0.25, rely = 0.1)

    # Criar botao resonsavel por aumentar a quantidade de dados
    btn_aumentar_dados = Tk.Button(root, text = "Aumentar dados", padx = 1, pady = 1, fg = "white", bg = "#00006F", command = aumentar_dados)
    btn_aumentar_dados.place(relwidth = 0.27, relheight = 0.8, relx = 0.46, rely = 0.1)

    # Criar botao resonsavel por aumentar a quantidade de dados
    btn_classificadores = Tk.Button(root, text = "Classificadores", padx = 1, pady = 1, fg = "white", bg = "#00006F", command = construir_menu_classificador)
    btn_classificadores.place(relwidth = 0.24, relheight = 0.8, relx = 0.74, rely = 0.1)

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

"""
Nome: construir_menu_classificador
Funcao: Criar menu com configuração reponsavel
por disponibilizar os classificadores utilizados.
"""
def construir_menu_classificador():
    # Configurar tela
    limpar_menu()
    root.title("Escolha o CLassificador");
    root.minsize(220, 50)
    root.maxsize(220, 50)
    screen = Tk.Canvas(root, height = 50, width= 220, bg = "#202020")
    screen.pack()

    # Criar botao com opcao para salvar
    btn_shufflenet = Tk.Button(root, text = "Shufflenet", padx = 1, pady = 1, fg = "white", bg = "#00006F", command = shufflenet)
    btn_shufflenet.place(relwidth = 0.4, relheight = 0.8, relx = 0.02, rely = 0.1)

construir_menu_principal() # Chamar primeira criacao/configuracao de tela
root.mainloop() # Deixar tela aberta