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
import glob
import sys
import tarfile
import time
import tkinter as Tk
import warnings
import scipy
from asyncio.windows_events import NULL
from logging import root
from tkinter import filedialog
import cv2
import matplotlib
import matplotlib.pyplot as plt
import tqdm
from numpy import asarray
from PIL import Image, ImageFilter
from matplotlib.pyplot import figure


# --- --- --- --- Declaracao do ShuffleNet --- --- --- ---
# Variaveis globais necessarias
root_path = './DataBase/.classifier'
root_path_train = './DataBase/train'
root_path_test = './DataBase/test/'
# Se o computador tiver cuda usar a GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#melhor acuracia de teste
best_acc = 0
start_epoch = 0
batch_size = 128
weight_decay = 5e-4
momentum = 0.9
learning_rate = 0.005
epoch_size = 60
LABEL_MAP = {0:'Nivel: 0', 1:'Nivel: 1', 2:'Nivel: 2', 3:'Nivel: 3', 4:'Nivel: 4'}

"""
Nome: ShuffleNet
Funcao: Implemtar aprendizado profundo na aplicacao, utilizando o
shufflenet como IA.
Descricao: Construcao do algoritmo shufflenet.
"""
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
        self.linear = nn.Linear(out_planes[2], 5) #10 as there are 10 classes

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

#Shufflenet com grupos = 2
net2 = ShuffleNetG2()

#Shufflenet com grupos = 3
net3 = ShuffleNetG3()

if torch.cuda.is_available():
  net3.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net3.parameters(), lr = learning_rate ,momentum= momentum, weight_decay=weight_decay)

def get_loss_acc(dataloader, is_test_dataset = True):
    net3.eval()
    # dataloader = test_loader if is_test_dataset else train-loader
    n_correct = 0
    n_total = 0
    test_loss = 0
    with torch.no_grad():
        for batch_size, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net3(inputs)
            test_loss += criterion(outputs, targets).item()
            _, predicted = outputs.max(1)
            n_correct +=predicted.eq(targets).sum().item()
            n_total += targets.shape[0]
    return test_loss/(batch_size+1), n_correct/n_total

def train_model(dataloader):
    net3.train()
    train_loss = 0
    n_correct = 0
    n_total = 0

    for batch_size, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net3(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        n_correct +=predicted.eq(targets).sum().item()
        n_total += targets.shape[0]
    return train_loss/(batch_size+1), n_correct/n_total

def save_best_model(epoch, dataloader):
    global best_acc
    net3.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net3(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct +=predicted.eq(targets).sum().item()
            print(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %0.3f%% (Correct classifications %d/Total classifications %d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

            #save checkpoint
            acc =100.*correct/total
            if acc>best_acc:
                print('Saving...')
                state = {'net': net3.state_dict(),
                         'acc': acc,
                         'epoch': epoch}
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, './checkpoint/ckpt.pth')
                torch.save(net3, './checkpoint/net3.pth')
                best_acc = acc

# --- --- --- --- Aplicacao --- --- --- ---

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
    fileNameFirst = filedialog.askopenfilename(
                    title = "Selecione a primeira imagem para comparação",
                    filetypes= extensionFiletypes)

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
        fileNameSecond = filedialog.askopenfilename(
                        title = "Selecione a segunda imagem para comparação",
                        filetypes= extensionFiletypes)

        secondImg = cv2.imread(fileNameSecond)

        # Rcomecar o menu caso nao exista imagem
        if secondImg is None:
            abrir_menu()

        # Fazer comparacao
        else:
            secondImg = cv2.cvtColor(secondImg, cv2.COLOR_BGR2GRAY)
            secondImg = cv2.equalizeHist(secondImg)

            # Comparar imagens
            resultComparation = cv2.matchTemplate(secondImg, firstImg,
                                                  cv2.TM_CCORR_NORMED)

            (vMin, vMax, cMin, cMax) = cv2.minMaxLoc(resultComparation)

            # Obter coordenadas do local de semelhanca da imagem
            (start_x, start_y) = cMax
            end_x = start_x + firstImg.shape[1]
            end_y = start_y + firstImg.shape[0]

            # Apresentar imagem com comparacao
            resultImage = cv2.imread(fileNameSecond)
            cv2.rectangle(resultImage, (start_x, start_y),
                          (end_x, end_y), (255, 0, 0), 2)
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
    filepathAndFile = filedialog.asksaveasfilename(
                    title = "Selecione a pasta para armazenar o arquivo",
                    filetypes = extensionFiletypes)

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
Funcao: Aplicar o aprendizado shufflenet.
"""
def shufflenet():
    transform_train = transforms.Compose(transforms=[transforms.Pad(4),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.Resize((50, 50)),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    transform_test = transforms.Compose(transforms=[transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    train_set = dsets.ImageFolder(root=root_path_train, transform=transform_train)
    test_set = dsets.ImageFolder(root=root_path_test, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    f = open("./dadosShufflenet.txt", "w")

    EPOCH = epoch_size
    training_time = dt.datetime.now()
    start = dt.datetime.now()
    start_epoch = 0
    train_accuracy_list = []
    test_accuracy_list = []

    print('INICIO DO TREINAMENTO')
    for epoch_i in range(start_epoch, start_epoch + EPOCH):
      global accuracy_list
      current_learning_rate = [i['lr'] for i in optimizer.param_groups][0]
      # Mostrar e arquivar dados de aprendizado
      f.write('Batch Size' + str(batch_size) + '(' + str((dt.datetime.now() - start).seconds) + ')\n\nEpoch: ' + str(epoch_i +1) + '/' + str(EPOCH+start_epoch) + ' | Current Learning Rate: ' + str(current_learning_rate))
      print('Batch Size', batch_size, '(%0.2fs)\n\nEpoch: %d/%d | Current Learning Rate: %.4f ' % ((dt.datetime.now() - start).seconds, epoch_i +1, EPOCH+start_epoch , current_learning_rate))

      start = dt.datetime.now()
      test_loss, test_acc = get_loss_acc(train_loader)
      train_loss, train_acc = train_model(train_loader)
      train_accuracy_list.append(train_acc*100)
      test_accuracy_list.append(test_acc*100)
      save_best_model(epoch_i, train_loader)

      # Mostrar e arquivar dados de aprendizado
      print('Train Loss: %.3f | Acc: %.3f%% \nTest Loss: %0.3f | Acc: %0.3f%% \n\n' % (train_loss, train_acc*100, test_loss, test_acc*100))
      f.write('Train Loss: ' + str(train_loss) + ' | Acc: ' + str(train_acc*100) + ' \nTest Loss: ' + str(test_loss) + ' | Acc: ' + str(test_acc*100) + ' \n\n')

    print('\n\nTotal Training time: %0.2f minutes ' %((dt.datetime.now() - training_time).seconds/60))
    f.write('\n\nTotal Training time: ' + str((dt.datetime.now() - training_time).seconds/60) + ' minutes ')
    f.close()


"""
Nome: apresentar_resultados
Funcao: Mostrar resultados do aprendizado da rede.
"""
def apresentar_resultados_shufflenet():
    tipo = 'cuda' if torch.cuda.is_available() else 'cpu'
    net3 = torch.load('./checkpoint/net3.pth', map_location=torch.device(tipo))

    transform_test = transforms.Compose(transforms=[transforms.Resize((50, 50)), transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    test_set = dsets.ImageFolder(root=root_path_test, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    #Model Accuracy
    total_correct = 0
    total_images = 0
    confusion_matrix = np.zeros([5,5], int)

    with torch.no_grad():
        for i,data in enumerate(test_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net3(images)
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            for i, l in enumerate(labels):
                confusion_matrix[l.item(), predicted[i].item()] +=1

    print('{0:5s} - {1}'.format('Category', 'Accuracy'))
    for i, r in enumerate(confusion_matrix):
      print('{0:5s} - {1:0.1f}'.format(LABEL_MAP[i], r[i]/np.sum(r)*100))
    model_accuracy = total_correct/total_images *100
    print('Model Accuracy on {0} test images : {1:.2f} %'.format(total_images, model_accuracy))

    fig, axis = plt.subplots(1,1,figsize = (5,5))
    axis.matshow(confusion_matrix, aspect='auto', vmin = 0, vmax = 1000, cmap= plt.get_cmap('Wistia'))
    for (i, j), z in np.ndenumerate(confusion_matrix):
        valor_linha = 0
        for index in range(0 , 5):
            valor_linha += confusion_matrix[i,index]
        axis.text(j, i, '{:0.2f}'.format(z*100/valor_linha), ha='center', va='center')
    plt.ylabel('Actual Category')
    plt.yticks(range(5), LABEL_MAP.values())
    plt.xlabel('Predicted Category')
    plt.xticks(range(5), LABEL_MAP.values())
    plt.rcParams.update({'font.size': 14})
    plt.show()

    classificar

"""
Nome: classificar_shufflenet
Funcao: Pegar uma imagem e classificala por meio do que foi aprendido.
"""
def classificar_shufflenet():
    tipo = 'cuda' if torch.cuda.is_available() else 'cpu'
    net3 = torch.load('./checkpoint/net3.pth', map_location=torch.device("cpu"))

    fileName = filedialog.askopenfilename(filetypes= (("PNG","*.png"), ("JPG","*.jpg")))
    imagem = Image.open(r""+fileName)
    nomePastaTotal = fileName.split('/')
    nomePasta = nomePastaTotal[len(nomePastaTotal)-2]
    path = root_path + '/' + nomePasta

    os.chdir(path)
    imagem.save(r""+"img.png")
    os.chdir('./../../..')

    transform_classifier = transforms.Compose(transforms=[transforms.Grayscale(num_output_channels=3),
                                              transforms.Resize((50, 50)), transforms.ToTensor()])
    outarray = transform_classifier(imagem)

    class_mapping = ["0", "1", "2", "3", "4"]

    net3.eval()
    with torch.no_grad():
        predictions = net3(torch.utils.data.DataLoader((outarray, nomePasta)))
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        print(predicted)

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
    btn_abrir_arquivo = Tk.Button(root, text = "Cortar Imagem", padx = 1,
                 pady = 1, fg = "white", bg = "#00006F", command = abrir_imagem)
    btn_abrir_arquivo.place(relwidth = 0.22, relheight = 0.8, relx = 0.02, rely = 0.1)

    # Criar botao resonsavel por selecionar imagens que serao comparadas
    btn_comparar = Tk.Button(root, text = "Comparar", padx = 1, pady = 1,
                               fg = "white", bg = "#00006F", command = comparar)
    btn_comparar.place(relwidth = 0.20, relheight = 0.8, relx = 0.25, rely = 0.1)

    # Criar botao resonsavel por aumentar a quantidade de dados
    btn_aumentar_dados = Tk.Button(root, text = "Aumentar dados", padx = 1,
                                   pady = 1, fg = "white", bg = "#00006F", command = aumentar_dados)
    btn_aumentar_dados.place(relwidth = 0.27, relheight = 0.8, relx = 0.46, rely = 0.1)

    # Criar botao resonsavel por aumentar a quantidade de dados
    btn_classificadores = Tk.Button(root, text = "Classificadores", padx = 1,
                                    pady = 1, fg = "white", bg = "#00006F", command = construir_menu_classificador)
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
    btn_salvar = Tk.Button(root, text = "Salvar Imagem", padx = 1, pady = 1,
                           fg = "white", bg = "GREEN", command = salvar_imagem_gerada)
    btn_salvar.place(relwidth = 0.4, relheight = 0.8, relx = 0.02, rely = 0.1)

    # Criar botao com opcao de nao salvar
    btn_nao_salvar = Tk.Button(root, text = "Não Salvar", padx = 1, pady = 1,
                               fg = "white", bg = "RED", command = nao_salvar_imagem_gerada)
    btn_nao_salvar.place(relwidth = 0.4, relheight = 0.8, relx = 0.585, rely = 0.1)
"""
Nome: construir_menu_shufflenet
Funcao: Criar menu com opcoes do classsificador.
"""
def construir_menu_shufflenet():
    # Configurar tela
    limpar_menu()
    root.title("Escolha a Opcao - ShuffleNet");
    root.minsize(380, 50)
    root.maxsize(380, 50)
    screen = Tk.Canvas(root, height = 50, width= 380, bg = "#202020")
    screen.pack()

    # Criar botao com opcao para salvar
    btn_shufflenet = Tk.Button(root, text = "Treinar", padx = 1, pady = 1,
                               fg = "white", bg = "#00006F", command = shufflenet)
    btn_shufflenet.place(relwidth = 0.25, relheight = 0.8, relx = 0.02, rely = 0.1)

    # Criar botao com opcao para salvar
    btn_shufflenet = Tk.Button(root, text = "Apresentar Resultados", padx = 1, pady = 1,
                               fg = "white", bg = "#00006F", command = apresentar_resultados_shufflenet)
    btn_shufflenet.place(relwidth = 0.44, relheight = 0.8, relx = 0.28, rely = 0.1)

    # Criar botao com opcao para salvar
    btn_shufflenet = Tk.Button(root, text = "Classificar", padx = 1, pady = 1,
                               fg = "white", bg = "#00006F", command = classificar_shufflenet)
    btn_shufflenet.place(relwidth = 0.25, relheight = 0.8, relx = 0.73, rely = 0.1)

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
    btn_shufflenet = Tk.Button(root, text = "Shufflenet", padx = 1, pady = 1,
                               fg = "white", bg = "#00006F", command = construir_menu_shufflenet)
    btn_shufflenet.place(relwidth = 0.4, relheight = 0.8, relx = 0.02, rely = 0.1)

construir_menu_principal() # Chamar primeira criacao/configuracao de tela
root.mainloop() # Deixar tela aberta

construir_menu_principal() # Chamar primeira criacao/configuracao de tela
root.mainloop() # Deixar tela aberta
