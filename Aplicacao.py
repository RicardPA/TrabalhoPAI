"""
Ciencias da Computação/Praça da Liberdade

Alunos:
* Lucas Satlher Campos Lacerda
* Ricardo Portilho de Andrade
"""

# Bibliotecas utilizadas no Aprendizado de Maquina
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torchvision import models
from torchsummary import summary
from torch import nn, optim
import torch.nn.functional as functions
import torchvision
from torch.utils.data import IterableDataset
import torchvision.transforms.functional as TF

# Tratamento de dados
import numpy as np
from numpy import asarray
#import pandas as pd

# Utilizado em classificadores
import sklearn
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing

# Criacao de telas
import tkinter as Tk
from tkinter import filedialog

# Manipulacao de arquivos e imagens
import os
import cv2
import glob
import pickle
from PIL import Image, ImageFilter

# Plotar graficos
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import datetime as dt
import warnings
import scipy
from asyncio.windows_events import NULL
from logging import root
import tqdm
import queue



# --- --- --- --- Declaracao do ShuffleNet --- --- --- ---
# Variaveis globais necessarias
root_path_train = './DataBase/train'
root_path_test = './DataBase/test'
root_path_val = './DataBase/val'
# Se o computador tiver cuda usar a GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#melhor acuracia de teste
weight_decay = 5e-4
learning_rate = 0.01
epoch_size = 60
best_acc = 0
start_epoch = 0
batch_size = 128
momentum = 0.9
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
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, groups=g, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.shuffle1 = ShuffleBlock(groups= g)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes, bias=True)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, groups=groups, bias=True)
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
        self.conv1 = nn.Conv2d(3, 24, kernel_size=1, bias = True)
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

            #save checkpoint
            acc =100.*correct/total
            if acc>best_acc:
                state = {'net': net3.state_dict(),
                         'acc': acc,
                         'epoch': epoch}
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, './checkpoint/ckpt.pth')
                torch.save(net3, './checkpoint/net3.pth')
                best_acc = acc

class MyDataset(IterableDataset):
    def __init__(self, image_queue):
        self.queue = image_queue

    def read_next_image(self):
        while self.queue.qsize() > 0:
            yield self.queue.get()
        return None

    def __iter__(self):
        return self.read_next_image()

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
    transform_train = transforms.Compose(transforms=[transforms.Resize((224, 224)),
                                         transforms.CenterCrop((90, 210)),
                                         transforms.RandomEqualize(p=1),
                                         transforms.Resize((50, 50)),
                                         transforms.ToTensor()])

    train_set = dsets.ImageFolder(root=root_path_train, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

    EPOCH = epoch_size
    training_time = dt.datetime.now()
    start = dt.datetime.now()
    start_epoch = 0
    train_accuracy_list = []
    test_accuracy_list = []

    for epoch_i in range(start_epoch, start_epoch + EPOCH):
      global accuracy_list
      current_learning_rate = [i['lr'] for i in optimizer.param_groups][0]

      start = dt.datetime.now()
      test_loss, test_acc = get_loss_acc(train_loader)
      train_loss, train_acc = train_model(train_loader)
      train_accuracy_list.append(train_acc*100)
      test_accuracy_list.append(test_acc*100)
      save_best_model(epoch_i, train_loader)

    apresentar_dados("FIM DO TREINAMENTO"+
    "\n* O modelo treinado foi armazenado no arquivo:\n\t (./checkpoint/net3.pth)"+
    '\n Tempo total de treinamento: ' + str(int((dt.datetime.now() - training_time).seconds/60)) + ' minutos ')

    f = open("./dados/shufflenet.sav", "w")
    f.write("")
    f.close()

"""
Nome: apresentar_resultados
Funcao: Mostrar resultados do aprendizado da rede.
"""
def apresentar_resultados_shufflenet():
    dados = {
        'info': '',
        'matriz': np.zeros([5,5], int)
    }
    tipo = 'cuda' if torch.cuda.is_available() else 'cpu'
    if os.path.exists('./dados/shufflenet.sav') and os.stat("./dados/shufflenet.sav").st_size != 0:
        dados = pickle.load(open('./dados/shufflenet.sav', 'rb'))
    else:
        net3 = torch.load('./checkpoint/net3.pth', map_location=torch.device(tipo))

        transform_test = transforms.Compose(transforms=[transforms.Resize((224, 224)),
                                                         transforms.CenterCrop((90, 210)),
                                                         transforms.RandomEqualize(p=1),
                                                         transforms.Resize((50, 50)),
                                                         transforms.ToTensor()])

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

        dados['matriz'] = confusion_matrix
        # Criar vetor para armazenar os valores da formular
        vp = np.zeros(5, int)
        vn = np.zeros(5, int)
        fp = np.zeros(5, int)
        fn = np.zeros(5, int)

        # pegar o valor para cada classe
        for i in range(0, 5):
            for j in range(0, 5):
                for k in range(0, 5):
                    if i == k and j == k:
                        vp[k] += confusion_matrix[i, j]

                    if i != k and j == k:
                        fp[k] += confusion_matrix[i, j]

                    if i == k and j != k:
                        fn[k] += confusion_matrix[i, j]

                    if i != k and j != k:
                        vn[k] += confusion_matrix[i, j]

        # Apresentar resultados do treino
        info = ('{0:5s} - {1:5s} - {2:5s} - {3:5s} - {4}'.format('Acuracia', 'Sensibilidade', 'Especificidade', 'Precisão', 'Score F1'))
        for i in range(0, 5):
            acuracia = ((vp[i] + vn[i])/(vp[i] + vn[i] + fp[i] + fn[i]))*100 if not isNaN(((vp[i] + vn[i])/(vp[i] + vn[i] + fp[i] + fn[i]))*100) else 0
            sensibilidade = (vp[i]/(vp[i] + fn[i]))*100 if not isNaN((vp[i]/(vp[i] + fn[i]))*100) else 0
            especificidade = (vn[i]/(vn[i] + fp[i]))*100 if not isNaN((vn[i]/(vn[i] + fp[i]))*100) else 0
            precisao = (vp[i]/(vp[i]+fp[i]))*100 if not isNaN((vp[i]/(vp[i]+fp[i]))*100) else 0
            scoref1 = ((2*vp[i])/((2*vp[i])+fp[i]+fn[i]))*100 if not isNaN(((2*vp[i])/((2*vp[i])+fp[i]+fn[i]))*100) else 0
            info += '\n' + ('{0:5s} - {1:0.1f}% - {2:0.1f}% - {3:0.1f}% - {4:0.1f}% - {5:0.1f}%'.format(LABEL_MAP[i], acuracia, sensibilidade, especificidade, precisao, scoref1))
        model_accuracy = total_correct/total_images *100
        info += '\n\n' + ('Acuracia do Modelo com {0} no teste : {1:.2f} %'.format(total_images, model_accuracy))
        dados['info'] = info
        pickle.dump(dados, open('./dados/shufflenet.sav', 'wb'))

    confusion_matrix = dados['matriz']
    apresentar_dados(dados['info'])
    # Mostrar matriz de confusao
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

    buffer = queue.Queue()

    transformacao = transforms.Compose(transforms=[transforms.Grayscale(num_output_channels=3),
                                                   transforms.Resize((224, 224)),
                                                   transforms.CenterCrop((90, 210)),
                                                   transforms.RandomEqualize(p=1),
                                                   transforms.Resize((50, 50))])

    buffer.put(TF.to_tensor(transformacao(imagem)))

    dataset = MyDataset(buffer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    class_mapping = ["0", "1", "2", "3", "4"]

    for data in dataloader:
        classifier_time = dt.datetime.now()
        predictions = net3(data)
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]

    resultado = 'Tem osteoartrite' if predict > 1 else 'Não tem osteoartrite'
    apresentar_dados("FIM DA CLASSIFICAÇÃO"+
    "Joelho cassificado como: " + str(predicted)+
    "\n Resultado: " + resultado +
    '\n Tempo total de clasificação: ' + str((dt.datetime.now() - classifier_time).seconds) + ' segundos ')

"""
Nome: SVM
Funcao: Implemtar aprendizado profundo na aplicacao, utilizando o
svm como IA.
Descricao: Construcao do algoritmo svm.
"""
def SVM():
    data = []
    target = []
    targetB = []

    for i, _, arquivos in os.walk(root_path_train):
        for arquivo in arquivos:
            i_aux = str(i).replace("\\", "/")
            nomePastaTotal = i_aux.split('/')
            nomePasta = nomePastaTotal[len(nomePastaTotal)-1]

            imagem = Image.open(r""+i_aux+'/'+arquivo)

            transformacao = transforms.Compose(transforms=[transforms.Grayscale(num_output_channels=3),
                                                      transforms.Resize((224, 224)),
                                                      transforms.CenterCrop((90, 220))])

            imagem = transformacao(imagem)

            imagem = np.array(imagem)

            gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

            gray = cv2.equalizeHist(gray)

            ret, thresh = cv2.threshold(gray, 0, 255,
                                        cv2.THRESH_BINARY_INV +
                                        cv2.THRESH_OTSU)

            kernel = np.ones((3, 3), np.uint8)

            closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                                        kernel, iterations = 2)

            bg = cv2.dilate(closing, kernel, iterations = 1)

            dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0)

            ret, imagem = cv2.threshold(dist_transform, 0.02
                                    * dist_transform.max(), 255, 0)

            data.append(imagem)
            target.append(nomePasta)
            classeB = 0 if int(nomePasta) < 2 else 1
            targetB.append(classeB)

    for i, _, arquivos in os.walk(root_path_test):
        for arquivo in arquivos:
            i_aux = str(i).replace("\\", "/")
            nomePastaTotal = i_aux.split('/')
            nomePasta = nomePastaTotal[len(nomePastaTotal)-1]

            imagem = Image.open(r""+i_aux+'/'+arquivo)

            transformacao = transforms.Compose(transforms=[transforms.Grayscale(num_output_channels=3),
                                                      transforms.Resize((224, 224)),
                                                      transforms.CenterCrop((90, 220))])

            imagem = transformacao(imagem)

            imagem = np.array(imagem)

            gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

            gray = cv2.equalizeHist(gray)

            ret, thresh = cv2.threshold(gray, 0, 255,
                                        cv2.THRESH_BINARY_INV +
                                        cv2.THRESH_OTSU)

            kernel = np.ones((3, 3), np.uint8)

            closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                                        kernel, iterations = 2)

            bg = cv2.dilate(closing, kernel, iterations = 1)

            dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0)

            ret, imagem = cv2.threshold(dist_transform, 0.02
                                    * dist_transform.max(), 255, 0)

            data.append(imagem)
            target.append(nomePasta)
            classeB = 0 if int(nomePasta) < 2 else 1
            targetB.append(classeB)

    X_train = np.array(data)
    nsamples = X_train.shape[0]
    nx = X_train.shape[1]
    ny = X_train.shape[2]
    X_train = X_train.reshape((nsamples,nx*ny))

    training_time = dt.datetime.now()
    clf = svm.SVC(kernel='poly', class_weight='balanced')
    clf.fit(X_train, target)
    training_time = dt.datetime.now() - training_time

    training_timeB = dt.datetime.now()
    clfB = svm.SVC(kernel='linear', class_weight='balanced')
    clfB.fit(X_train, targetB)

    apresentar_dados("FIM DO TREINAMENTO"+
    "\n* O modelo treinado foi armazenado no arquivo:\n\t(./checkpoint/smv.sav) \n\t(./checkpoint/smvB.sav)"+
    '\n\n Tempo total de treinamento do modelo multiclasse: ' + str(int((training_time).seconds/60)) + ' minutos '+
    '\n Tempo total de treinamento do modelo binário: ' + str(int((dt.datetime.now() - training_timeB).seconds/60)) + ' minutos ')

    pickle.dump(clf, open('./checkpoint/svm.sav', 'wb'))
    pickle.dump(clfB, open('./checkpoint/svmB.sav', 'wb'))

    f = open("./dados/svm.sav", "w")
    f.write("")
    f.close()

    f = open("./dados/svmB.sav", "w")
    f.write("")
    f.close()

def apresentar_resultados_svm():
    data = []
    target = []
    dados = {
        'info': '',
        'matriz': np.zeros([5,5], int)
    }

    if os.path.exists('./dados/svm.sav') and os.stat("./dados/svm.sav").st_size != 0:
        dados = pickle.load(open('./dados/svm.sav', 'rb'))
    else:
        for i, _, arquivos in os.walk(root_path_val):
            for arquivo in arquivos:
                i_aux = str(i).replace("\\", "/")
                nomePastaTotal = i_aux.split('/')
                nomePasta = nomePastaTotal[len(nomePastaTotal)-1]

                imagem = Image.open(r""+i_aux+'/'+arquivo)

                transformacao = transforms.Compose(transforms=[transforms.Grayscale(num_output_channels=3),
                                                          transforms.Resize((224, 224)),
                                                          transforms.CenterCrop((90, 220))])

                imagem = transformacao(imagem)

                imagem = np.array(imagem)

                gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

                gray = cv2.equalizeHist(gray)

                ret, thresh = cv2.threshold(gray, 0, 255,
                                            cv2.THRESH_BINARY_INV +
                                            cv2.THRESH_OTSU)

                kernel = np.ones((3, 3), np.uint8)

                closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                                            kernel, iterations = 2)

                bg = cv2.dilate(closing, kernel, iterations = 1)

                dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0)

                ret, imagem = cv2.threshold(dist_transform, 0.02
                                        * dist_transform.max(), 255, 0)

                data.append(imagem)
                target.append(nomePasta)

        X_test = np.array(data)
        nsamples = X_test.shape[0]
        nx = X_test.shape[1]
        ny = X_test.shape[2]
        X_test = X_test.reshape((nsamples,nx*ny))

        clf = pickle.load(open('./checkpoint/svm.sav', 'rb'))
        y_pred = clf.predict(X_test)

        # Criar vetor para armazenar os valores da formular
        vp = np.zeros(5, int)
        vn = np.zeros(5, int)
        fp = np.zeros(5, int)
        fn = np.zeros(5, int)
        confusion_matrix = np.zeros([5,5], int)

        for i in range(0, len(target)):
            x = int(target[i])
            y = int(y_pred[i])
            confusion_matrix[x, y] = confusion_matrix[x, y] + 1

        dados['matriz'] = confusion_matrix

        # pegar o valor para cada classe
        for i in range(0, 5):
            for j in range(0, 5):
                for k in range(0, 5):
                    if i == k and j == k:
                        vp[k] += confusion_matrix[i, j]

                    if i != k and j == k:
                        fp[k] += confusion_matrix[i, j]

                    if i == k and j != k:
                        fn[k] += confusion_matrix[i, j]

                    if i != k and j != k:
                        vn[k] += confusion_matrix[i, j]

        # Apresentar resultados do treino
        info = ('{0:5s} - {1:5s} - {2:5s} - {3:5s} - {4}'.format('Acuracia', 'Sensibilidade', 'Especificidade', 'Precisão', 'Score F1'))
        for i in range(0, 5):
            acuracia = ((vp[i] + vn[i])/(vp[i] + vn[i] + fp[i] + fn[i]))*100 if not isNaN(((vp[i] + vn[i])/(vp[i] + vn[i] + fp[i] + fn[i]))*100) else 0
            sensibilidade = (vp[i]/(vp[i] + fn[i]))*100 if not isNaN((vp[i]/(vp[i] + fn[i]))*100) else 0
            especificidade = (vn[i]/(vn[i] + fp[i]))*100 if not isNaN((vn[i]/(vn[i] + fp[i]))*100) else 0
            precisao = (vp[i]/(vp[i]+fp[i]))*100 if not isNaN((vp[i]/(vp[i]+fp[i]))*100) else 0
            scoref1 = ((2*vp[i])/((2*vp[i])+fp[i]+fn[i]))*100 if not isNaN(((2*vp[i])/((2*vp[i])+fp[i]+fn[i]))*100) else 0
            info += '\n' + ('{0:5s} - {1:0.1f}% - {2:0.1f}% - {3:0.1f}% - {4:0.1f}% - {5:0.1f}%'.format(LABEL_MAP[i], acuracia, sensibilidade, especificidade, precisao, scoref1))
        model_accuracy = ((np.sum(vp) + np.sum(vn))/(np.sum(vp) + np.sum(vn) + np.sum(fp) + np.sum(fn))) * 100
        info += '\n\n' + ('Acuracia do Modelo com {0} no teste : {1:.2f} %'.format(len(data), model_accuracy))

        dados['info'] = info
        pickle.dump(dados, open('./dados/svm.sav', 'wb'))

    # Mostrar matriz de confusao
    fig, axis = plt.subplots(1,1,figsize = (5,5))
    axis.matshow(dados['matriz'], aspect='auto', vmin = 0, vmax = 1000, cmap= plt.get_cmap('Wistia'))
    for (i, j), z in np.ndenumerate(dados['matriz']):
        valor_linha = 0
        for index in range(0 , 5):
            valor_linha += dados['matriz'][i,index]
        axis.text(j, i, '{:0.2f}'.format(z*100/valor_linha), ha='center', va='center')
    plt.ylabel('Actual Category')
    plt.yticks(range(5), LABEL_MAP.values())
    plt.xlabel('Predicted Category')
    plt.xticks(range(5), LABEL_MAP.values())
    plt.rcParams.update({'font.size': 14})
    apresentar_dados(dados['info'])
    plt.show()

def apresentar_resultados_svmB():
    data = []
    target = []
    dados = {
        'info': '',
        'matriz': np.zeros([2,2], int)
    }

    if os.path.exists('./dados/svmB.sav') and os.stat("./dados/svmB.sav").st_size != 0:
        dados = pickle.load(open('./dados/svmB.sav', 'rb'))
    else:
        for i, _, arquivos in os.walk(root_path_val):
            for arquivo in arquivos:
                i_aux = str(i).replace("\\", "/")
                nomePastaTotal = i_aux.split('/')
                nomePasta = nomePastaTotal[len(nomePastaTotal)-1]

                imagem = Image.open(r""+i_aux+'/'+arquivo)

                transformacao = transforms.Compose(transforms=[transforms.Grayscale(num_output_channels=3),
                                                          transforms.Resize((224, 224)),
                                                          transforms.CenterCrop((90, 220))])

                imagem = transformacao(imagem)

                imagem = np.array(imagem)

                gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

                gray = cv2.equalizeHist(gray)

                ret, thresh = cv2.threshold(gray, 0, 255,
                                            cv2.THRESH_BINARY_INV +
                                            cv2.THRESH_OTSU)

                kernel = np.ones((3, 3), np.uint8)

                closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                                            kernel, iterations = 2)

                bg = cv2.dilate(closing, kernel, iterations = 1)

                dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0)

                ret, imagem = cv2.threshold(dist_transform, 0.02
                                        * dist_transform.max(), 255, 0)

                data.append(imagem)
                classe = 0 if int(nomePasta) < 2 else 1
                target.append(classe)


        X_test = np.array(data)
        nsamples = X_test.shape[0]
        nx = X_test.shape[1]
        ny = X_test.shape[2]
        X_test = X_test.reshape((nsamples,nx*ny))

        clf = pickle.load(open('./checkpoint/svmB.sav', 'rb'))
        y_pred = clf.predict(X_test)

        # Criar vetor para armazenar os valores da formular
        vp = np.zeros(2, int)
        vn = np.zeros(2, int)
        fp = np.zeros(2, int)
        fn = np.zeros(2, int)
        confusion_matrix = np.zeros([2,2], int)

        for i in range(0, len(target)):
            x = int(target[i])
            y = int(y_pred[i])
            confusion_matrix[x, y] = confusion_matrix[x, y] + 1

        dados['matriz'] = confusion_matrix

        # pegar o valor para cada classe
        for i in range(0, 2):
            for j in range(0, 2):
                for k in range(0, 2):
                    if i == k and j == k:
                        vp[k] += confusion_matrix[i, j]

                    if i != k and j == k:
                        fp[k] += confusion_matrix[i, j]

                    if i == k and j != k:
                        fn[k] += confusion_matrix[i, j]

                    if i != k and j != k:
                        vn[k] += confusion_matrix[i, j]

        # Apresentar resultados do treino
        info = ('{0:5s} - {1:5s} - {2:5s} - {3:5s} - {4}'.format('Acuracia', 'Sensibilidade', 'Especificidade', 'Precisão', 'Score F1'))
        for i in range(0, 2):
            acuracia = ((vp[i] + vn[i])/(vp[i] + vn[i] + fp[i] + fn[i]))*100 if not isNaN(((vp[i] + vn[i])/(vp[i] + vn[i] + fp[i] + fn[i]))*100) else 0
            sensibilidade = (vp[i]/(vp[i] + fn[i]))*100 if not isNaN((vp[i]/(vp[i] + fn[i]))*100) else 0
            especificidade = (vn[i]/(vn[i] + fp[i]))*100 if not isNaN((vn[i]/(vn[i] + fp[i]))*100) else 0
            precisao = (vp[i]/(vp[i]+fp[i]))*100 if not isNaN((vp[i]/(vp[i]+fp[i]))*100) else 0
            scoref1 = ((2*vp[i])/((2*vp[i])+fp[i]+fn[i]))*100 if not isNaN(((2*vp[i])/((2*vp[i])+fp[i]+fn[i]))*100) else 0
            info += '\n' + ('{0:5s} - {1:0.1f}% - {2:0.1f}% - {3:0.1f}% - {4:0.1f}% - {5:0.1f}%'.format(LABEL_MAP[i], acuracia, sensibilidade, especificidade, precisao, scoref1))
        model_accuracy = ((np.sum(vp) + np.sum(vn))/(np.sum(vp) + np.sum(vn) + np.sum(fp) + np.sum(fn))) * 100
        info += '\n\n' + ('Acuracia do Modelo com {0} no teste : {1:.2f} %'.format(len(data), model_accuracy))

        dados['info'] = info
        pickle.dump(dados, open('./dados/svmB.sav', 'wb'))

    # Mostrar matriz de confusao
    fig, axis = plt.subplots(1,1,figsize = (2,2))
    axis.matshow(dados['matriz'], aspect='auto', vmin = 0, vmax = 1000, cmap= plt.get_cmap('Wistia'))
    for (i, j), z in np.ndenumerate(dados['matriz']):
        valor_linha = 0
        for index in range(0 , 2):
            valor_linha += dados['matriz'][i,index]
        axis.text(j, i, '{:0.2f}'.format(z*100/valor_linha), ha='center', va='center')
    plt.ylabel('Actual Category')
    plt.yticks(range(2), {0: '0', 1: '1'}.values())
    plt.xlabel('Predicted Category')
    plt.xticks(range(2), {0: '0', 1: '1'}.values())
    plt.rcParams.update({'font.size': 14})
    apresentar_dados(dados['info'])
    plt.show()

def classificar_svm():
    data = []

    fileName = filedialog.askopenfilename(filetypes= (("PNG","*.png"), ("JPG","*.jpg")))
    nomePastaTotal = fileName.split('/')
    nomePasta = nomePastaTotal[len(nomePastaTotal)-2]

    imagem = Image.open(r""+fileName)

    transformacao = transforms.Compose(transforms=[transforms.Grayscale(num_output_channels=3),
                                              transforms.Resize((224, 224)),
                                              transforms.CenterCrop((90, 220))])

    imagem = transformacao(imagem)

    imagem = np.array(imagem)

    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    gray = cv2.equalizeHist(gray)

    ret, thresh = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY_INV +
                                cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)

    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                                kernel, iterations = 2)

    bg = cv2.dilate(closing, kernel, iterations = 1)

    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0)

    ret, imagem = cv2.threshold(dist_transform, 0.02
                            * dist_transform.max(), 255, 0)

    data.append(imagem)

    X_test = np.array(data)
    nsamples = X_test.shape[0]
    nx = X_test.shape[1]
    ny = X_test.shape[2]
    X_test = X_test.reshape((nsamples,nx*ny))

    clf = pickle.load(open('./checkpoint/svm.sav', 'rb'))
    classifier_time = dt.datetime.now()

    y_pred = clf.predict(X_test)

    resultado = 'Tem osteoartrite' if y_pred[0] > 1 else 'Não tem osteoartrite'
    apresentar_dados("FIM DA CLASSIFICAÇÃO"+
    "Joelho cassificado como: " + str(y_pred[0])+
    "\n Resultado: " + resultado)

def classificar_svmB():
    data = []

    fileName = filedialog.askopenfilename(filetypes= (("PNG","*.png"), ("JPG","*.jpg")))
    nomePastaTotal = fileName.split('/')
    nomePasta = nomePastaTotal[len(nomePastaTotal)-2]

    imagem = Image.open(r""+fileName)

    transformacao = transforms.Compose(transforms=[transforms.Grayscale(num_output_channels=3),
                                              transforms.Resize((224, 224)),
                                              transforms.CenterCrop((90, 220))])

    imagem = transformacao(imagem)

    imagem = np.array(imagem)

    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    gray = cv2.equalizeHist(gray)

    ret, thresh = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY_INV +
                                cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)

    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                                kernel, iterations = 2)

    bg = cv2.dilate(closing, kernel, iterations = 1)

    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0)

    ret, imagem = cv2.threshold(dist_transform, 0.02
                            * dist_transform.max(), 255, 0)

    data.append(imagem)

    X_test = np.array(data)
    nsamples = X_test.shape[0]
    nx = X_test.shape[1]
    ny = X_test.shape[2]
    X_test = X_test.reshape((nsamples,nx*ny))

    clf = pickle.load(open('./checkpoint/svmB.sav', 'rb'))
    classifier_time = dt.datetime.now()

    y_pred = clf.predict(X_test)

    resultado = 'Tem osteoartrite' if y_pred[0] == 1 else 'Não tem osteoartrite'
    apresentar_dados("FIM DA CLASSIFICAÇÃO"+
    "Joelho cassificado como: " + str(y_pred[0])+
    "\n Resultado: " + resultado)

"""
Nome: XGBoost
Funcao: Implemtar aprendizado profundo na aplicacao, utilizando o
XGBoost como IA.
Descricao: Construcao do algoritmo XGBoost.
"""
def XGBoost():
    data = []
    target = []
    targetB = []
    imagem = []

    for i, _, arquivos in os.walk(root_path_train):
        for arquivo in arquivos:
            i_aux = str(i).replace("\\", "/")
            nomePastaTotal = i_aux.split('/')
            nomePasta = nomePastaTotal[len(nomePastaTotal)-1]

            imagem = Image.open(r""+i_aux+'/'+arquivo)

            transformacao = transforms.Compose(transforms=[transforms.Resize((224, 224)),
                                                      transforms.CenterCrop((90, 220)),
                                                      transforms.Grayscale(num_output_channels=3)])

            imagem = transformacao(imagem)

            imagem = np.array(imagem)

            gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

            gray = cv2.equalizeHist(gray)

            ret, thresh = cv2.threshold(gray, 0, 255,
                                        cv2.THRESH_BINARY_INV +
                                        cv2.THRESH_OTSU)

            kernel = np.ones((3, 3), np.uint8)

            closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                                        kernel, iterations = 2)

            bg = cv2.dilate(closing, kernel, iterations = 1)

            dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0)

            ret, imagem = cv2.threshold(dist_transform, 0.02
                                    * dist_transform.max(), 255, 0)

            data.append(np.array(imagem).flatten())
            target.append(nomePasta)
            classeB = 0 if int(nomePasta) < 2 else 1
            targetB.append(classeB)

    for i, _, arquivos in os.walk(root_path_test):
        for arquivo in arquivos:
            i_aux = str(i).replace("\\", "/")
            nomePastaTotal = i_aux.split('/')
            nomePasta = nomePastaTotal[len(nomePastaTotal)-1]

            imagem = Image.open(r""+i_aux+'/'+arquivo)

            transformacao = transforms.Compose(transforms=[transforms.Grayscale(num_output_channels=3),
                                                      transforms.Resize((224, 224)),
                                                      transforms.CenterCrop((90, 220))])

            imagem = transformacao(imagem)

            imagem = np.array(imagem)

            gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

            gray = cv2.equalizeHist(gray)

            ret, thresh = cv2.threshold(gray, 0, 255,
                                        cv2.THRESH_BINARY_INV +
                                        cv2.THRESH_OTSU)

            kernel = np.ones((3, 3), np.uint8)

            closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                                        kernel, iterations = 2)

            bg = cv2.dilate(closing, kernel, iterations = 1)

            dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0)

            ret, imagem = cv2.threshold(dist_transform, 0.02
                                    * dist_transform.max(), 255, 0)

            data.append(np.array(imagem).flatten())
            target.append(nomePasta)
            classeB = 0 if int(nomePasta) < 2 else 1
            targetB.append(classeB)

    X_train = np.array(data)
    y_train = np.array(target)
    y_trainB = np.array(targetB)

    y_train = y_train.astype('int')

    training_time = dt.datetime.now()
    clf = XGBClassifier(learning_rate=0.1,
                        objective='multi:softmax',
                        booter='gbtree',
                        num_class=5,
                        max_depth=10)
    clf.fit(data, y_train)
    training_time = dt.datetime.now() - training_time

    training_timeB = dt.datetime.now()
    clfB = XGBClassifier(learning_rate=0.1,
                         objective='reg:linear',
                         booter='gbtree',
                         max_depth=10)
    clfB.fit(data, y_trainB)

    apresentar_dados("FIM DO TREINAMENTO"+
    "\n* O modelo treinado foi armazenado nos arquivos:\n\t(./checkpoint/xgboost.sav) \n\t(./checkpoint/xgboostB.sav)"+
    '\n\n Tempo total de treinamento do modelo multiclasse: ' + str(int((training_time).seconds/60)) + ' minutos '+
    '\n Tempo total de treinamento do modelo binário: ' + str(int((dt.datetime.now() - training_timeB).seconds/60)) + ' minutos ')

    pickle.dump(clf, open('./checkpoint/xgboost.sav', 'wb'))
    pickle.dump(clfB, open('./checkpoint/xgboostB.sav', 'wb'))

    f = open("./dados/xgboost.sav", "w")
    f.write("")
    f.close()

    f = open("./dados/xgboostB.sav", "w")
    f.write("")
    f.close()

def apresentar_resultados_xgboost():
    data = []
    target = []
    dados = {
        'info': '',
        'matriz': np.zeros([5,5], int),
    }

    if os.path.exists('./dados/xgboost.sav') and os.stat("./dados/xgboost.sav").st_size != 0:
        dados = pickle.load(open('./dados/xgboost.sav', 'rb'))

    else:
        for i, _, arquivos in os.walk(root_path_val):
            for arquivo in arquivos:
                i_aux = str(i).replace("\\", "/")
                nomePastaTotal = i_aux.split('/')
                nomePasta = nomePastaTotal[len(nomePastaTotal)-1]

                imagem = Image.open(r""+i_aux+'/'+arquivo)

                transformacao = transforms.Compose(transforms=[transforms.Grayscale(num_output_channels=3),
                                                          transforms.Resize((224, 224)),
                                                          transforms.CenterCrop((90, 220))])

                imagem = transformacao(imagem)

                imagem = np.array(imagem)

                gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

                gray = cv2.equalizeHist(gray)

                ret, thresh = cv2.threshold(gray, 0, 255,
                                            cv2.THRESH_BINARY_INV +
                                            cv2.THRESH_OTSU)

                kernel = np.ones((3, 3), np.uint8)

                closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                                            kernel, iterations = 2)

                bg = cv2.dilate(closing, kernel, iterations = 1)

                dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0)

                ret, imagem = cv2.threshold(dist_transform, 0.02
                                        * dist_transform.max(), 255, 0)

                data.append(np.array(imagem).flatten())
                target.append(nomePasta)

        clf = pickle.load(open('./checkpoint/xgboost.sav', 'rb'))

        y_pred = clf.predict(data)

        # Criar vetor para armazenar os valores da formular
        vp = np.zeros(5, int)
        vn = np.zeros(5, int)
        fp = np.zeros(5, int)
        fn = np.zeros(5, int)

        confusion_matrix = np.zeros([5,5], int)

        for i in range(0, len(target)):
            x = int(target[i])
            y = int(y_pred[i])
            confusion_matrix[x, y] = confusion_matrix[x, y] + 1

        dados['matriz'] = confusion_matrix

        # pegar o valor para cada classe
        for i in range(0, 5):
            for j in range(0, 5):
                for k in range(0, 5):
                    if i == k and j == k:
                        vp[k] += confusion_matrix[i, j]

                    if i != k and j == k:
                        fp[k] += confusion_matrix[i, j]

                    if i == k and j != k:
                        fn[k] += confusion_matrix[i, j]

                    if i != k and j != k:
                        vn[k] += confusion_matrix[i, j]

        # Apresentar resultados do treino
        info = ('{0:5s} - {1:5s} - {2:5s} - {3:5s} - {4}'.format('Acuracia', 'Sensibilidade', 'Especificidade', 'Precisão', 'Score F1'))
        for i in range(0, 5):
            acuracia = ((vp[i] + vn[i])/(vp[i] + vn[i] + fp[i] + fn[i]))*100 if not isNaN(((vp[i] + vn[i])/(vp[i] + vn[i] + fp[i] + fn[i]))*100) else 0
            sensibilidade = (vp[i]/(vp[i] + fn[i]))*100 if not isNaN((vp[i]/(vp[i] + fn[i]))*100) else 0
            especificidade = (vn[i]/(vn[i] + fp[i]))*100 if not isNaN((vn[i]/(vn[i] + fp[i]))*100) else 0
            precisao = (vp[i]/(vp[i]+fp[i]))*100 if not isNaN((vp[i]/(vp[i]+fp[i]))*100) else 0
            scoref1 = ((2*vp[i])/((2*vp[i])+fp[i]+fn[i]))*100 if not isNaN(((2*vp[i])/((2*vp[i])+fp[i]+fn[i]))*100) else 0
            info += '\n' + ('{0:5s} - {1:0.1f}% - {2:0.1f}% - {3:0.1f}% - {4:0.1f}% - {5:0.1f}%'.format(LABEL_MAP[i], acuracia, sensibilidade, especificidade, precisao, scoref1))
        model_accuracy = ((np.sum(vp) + np.sum(vn))/(np.sum(vp) + np.sum(vn) + np.sum(fp) + np.sum(fn))) * 100
        info += '\n\n' + ('Acuracia do Modelo com {0} no teste : {1:.2f} %'.format(len(data), model_accuracy))

        dados['info'] = info
        pickle.dump(dados, open('./dados/xgboost.sav', 'wb'))

    # Mostrar matriz de confusao
    fig, axis = plt.subplots(1,1,figsize = (5,5))
    axis.matshow(dados['matriz'], aspect='auto', vmin = 0, vmax = 1000, cmap= plt.get_cmap('Wistia'))
    for (i, j), z in np.ndenumerate(dados['matriz']):
        valor_linha = 0
        for index in range(0 , 5):
            valor_linha += dados['matriz'][i,index]
        axis.text(j, i, '{:0.2f}'.format(z*100/valor_linha), ha='center', va='center')
    plt.ylabel('Actual Category')
    plt.yticks(range(5), LABEL_MAP.values())
    plt.xlabel('Predicted Category')
    plt.xticks(range(5), LABEL_MAP.values())
    plt.rcParams.update({'font.size': 14})
    apresentar_dados(dados['info'])
    plt.show()

def apresentar_resultados_xgboostB():
    data = []
    target = []
    dados = {
        'info': '',
        'matriz': np.zeros([2,2], int),
    }

    if os.path.exists('./dados/xgboostB.sav') and os.stat("./dados/xgboostB.sav").st_size != 0:
        dados = pickle.load(open('./dados/xgboostB.sav', 'rb'))

    else:
        for i, _, arquivos in os.walk(root_path_val):
            for arquivo in arquivos:
                i_aux = str(i).replace("\\", "/")
                nomePastaTotal = i_aux.split('/')
                nomePasta = nomePastaTotal[len(nomePastaTotal)-1]

                imagem = Image.open(r""+i_aux+'/'+arquivo)

                transformacao = transforms.Compose(transforms=[transforms.Grayscale(num_output_channels=3),
                                                          transforms.Resize((224, 224)),
                                                          transforms.CenterCrop((90, 220))])

                imagem = transformacao(imagem)

                imagem = np.array(imagem)

                gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

                gray = cv2.equalizeHist(gray)

                ret, thresh = cv2.threshold(gray, 0, 255,
                                            cv2.THRESH_BINARY_INV +
                                            cv2.THRESH_OTSU)

                kernel = np.ones((3, 3), np.uint8)

                closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                                            kernel, iterations = 2)

                bg = cv2.dilate(closing, kernel, iterations = 1)

                dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0)

                ret, imagem = cv2.threshold(dist_transform, 0.02
                                        * dist_transform.max(), 255, 0)

                data.append(np.array(imagem).flatten())
                classe = 0 if int(nomePasta) < 2 else 1
                target.append(classe)

        clf = pickle.load(open('./checkpoint/xgboostB.sav', 'rb'))

        y_pred = clf.predict(data)

        # Criar vetor para armazenar os valores da formular
        vp = np.zeros(2, int)
        vn = np.zeros(2, int)
        fp = np.zeros(2, int)
        fn = np.zeros(2, int)

        confusion_matrix = np.zeros([2,2], int)

        for i in range(0, len(target)):
            x = int(target[i])
            y = int(y_pred[i])
            confusion_matrix[x, y] = confusion_matrix[x, y] + 1

        dados['matriz'] = confusion_matrix

        # pegar o valor para cada classe
        for i in range(0, 2):
            for j in range(0, 2):
                for k in range(0, 2):
                    if i == k and j == k:
                        vp[k] += confusion_matrix[i, j]

                    if i != k and j == k:
                        fp[k] += confusion_matrix[i, j]

                    if i == k and j != k:
                        fn[k] += confusion_matrix[i, j]

                    if i != k and j != k:
                        vn[k] += confusion_matrix[i, j]

        # Apresentar resultados do treino
        info = ('{0:5s} - {1:5s} - {2:5s} - {3:5s} - {4}'.format('Acuracia', 'Sensibilidade', 'Especificidade', 'Precisão', 'Score F1'))
        for i in range(0, 2):
            acuracia = ((vp[i] + vn[i])/(vp[i] + vn[i] + fp[i] + fn[i]))*100 if not isNaN(((vp[i] + vn[i])/(vp[i] + vn[i] + fp[i] + fn[i]))*100) else 0
            sensibilidade = (vp[i]/(vp[i] + fn[i]))*100 if not isNaN((vp[i]/(vp[i] + fn[i]))*100) else 0
            especificidade = (vn[i]/(vn[i] + fp[i]))*100 if not isNaN((vn[i]/(vn[i] + fp[i]))*100) else 0
            precisao = (vp[i]/(vp[i]+fp[i]))*100 if not isNaN((vp[i]/(vp[i]+fp[i]))*100) else 0
            scoref1 = ((2*vp[i])/((2*vp[i])+fp[i]+fn[i]))*100 if not isNaN(((2*vp[i])/((2*vp[i])+fp[i]+fn[i]))*100) else 0
            info += '\n' + ('{0:5s} - {1:0.1f}% - {2:0.1f}% - {3:0.1f}% - {4:0.1f}% - {5:0.1f}%'.format(LABEL_MAP[i], acuracia, sensibilidade, especificidade, precisao, scoref1))
        model_accuracy = ((np.sum(vp) + np.sum(vn))/(np.sum(vp) + np.sum(vn) + np.sum(fp) + np.sum(fn))) * 100
        info += '\n\n' + ('Acuracia do Modelo com {0} no teste : {1:.2f} %'.format(len(data), model_accuracy))

        dados['info'] = info
        pickle.dump(dados, open('./dados/xgboostB.sav', 'wb'))

    # Mostrar matriz de confusao
    fig, axis = plt.subplots(1,1,figsize = (2,2))
    axis.matshow(dados['matriz'], aspect='auto', vmin = 0, vmax = 1000, cmap= plt.get_cmap('Wistia'))
    for (i, j), z in np.ndenumerate(dados['matriz']):
        valor_linha = 0
        for index in range(0 , 2):
            valor_linha += dados['matriz'][i,index]
        axis.text(j, i, '{:0.2f}'.format(z*100/valor_linha), ha='center', va='center')
    plt.ylabel('Actual Category')
    plt.yticks(range(2), {0: '0', 1: '1'}.values())
    plt.xlabel('Predicted Category')
    plt.xticks(range(2), {0: '0', 1: '1'}.values())
    plt.rcParams.update({'font.size': 14})
    apresentar_dados(dados['info'])
    plt.show()

def classificar_xgboost():
    data = []

    fileName = filedialog.askopenfilename(filetypes= (("PNG","*.png"), ("JPG","*.jpg")))
    nomePastaTotal = fileName.split('/')
    nomePasta = nomePastaTotal[len(nomePastaTotal)-2]

    imagem = Image.open(r""+fileName)

    transformacao = transforms.Compose(transforms=[transforms.Grayscale(num_output_channels=3),
                                              transforms.Resize((224, 224)),
                                              transforms.CenterCrop((90, 220))])

    imagem = transformacao(imagem)

    imagem = np.array(imagem)

    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    gray = cv2.equalizeHist(gray)

    ret, thresh = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY_INV +
                                cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)

    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                                kernel, iterations = 2)

    bg = cv2.dilate(closing, kernel, iterations = 1)

    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0)

    ret, imagem = cv2.threshold(dist_transform, 0.02
                            * dist_transform.max(), 255, 0)

    data.append(np.array(imagem).flatten())

    clf = pickle.load(open('./checkpoint/xgboost.sav', 'rb'))
    y_pred = clf.predict(data)

    resultado = 'Tem osteoartrite' if y_pred[0] > 1 else 'Não tem osteoartrite'
    apresentar_dados("FIM DA CLASSIFICAÇÃO"+
    "\n Joelho cassificado como: " + str(y_pred[0])+
    '\n Resultado: ' + resultado)

def classificar_xgboostB():
    data = []

    fileName = filedialog.askopenfilename(filetypes= (("PNG","*.png"), ("JPG","*.jpg")))
    nomePastaTotal = fileName.split('/')
    nomePasta = nomePastaTotal[len(nomePastaTotal)-2]

    imagem = Image.open(r""+fileName)

    transformacao = transforms.Compose(transforms=[transforms.Grayscale(num_output_channels=3),
                                              transforms.Resize((224, 224)),
                                              transforms.CenterCrop((90, 220))])

    imagem = transformacao(imagem)

    imagem = np.array(imagem)

    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    gray = cv2.equalizeHist(gray)

    ret, thresh = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY_INV +
                                cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)

    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                                kernel, iterations = 2)

    bg = cv2.dilate(closing, kernel, iterations = 1)

    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0)

    ret, imagem = cv2.threshold(dist_transform, 0.02
                            * dist_transform.max(), 255, 0)

    data.append(np.array(imagem).flatten())

    clf = pickle.load(open('./checkpoint/xgboostB.sav', 'rb'))
    y_pred = clf.predict(data)

    resultado = 'Tem osteoartrite' if y_pred[0] == 1 else 'Não tem osteoartrite'
    apresentar_dados("FIM DA CLASSIFICAÇÃO"+
    "\n Joelho cassificado como: " + str(y_pred[0])+
    '\n Resultado: ' + resultado)

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
    root.title("Menu (Trabalho de PAI)")
    root.minsize(450, 500)
    root.maxsize(450, 500)
    screen = Tk.Canvas(root, height = 500, width = 450, bg = "#202020")
    screen.pack()

    # Criar botao responsavel por chamar o corte de uma imagem
    btn_abrir_arquivo = Tk.Button(root, text = "Cortar Imagem", padx = 1,
                 pady = 1, fg = "white", bg = "#00006F", command = abrir_imagem)
    btn_abrir_arquivo.place(relwidth = 0.22, relheight = 0.1, relx = 0.02, rely = 0.89)

    # Criar botao resonsavel por selecionar imagens que serao comparadas
    btn_comparar = Tk.Button(root, text = "Comparar", padx = 1, pady = 1,
                               fg = "white", bg = "#00006F", command = comparar)
    btn_comparar.place(relwidth = 0.20, relheight = 0.1, relx = 0.25, rely = 0.89)

    # Criar botao resonsavel por aumentar a quantidade de dados
    btn_aumentar_dados = Tk.Button(root, text = "Aumentar dados", padx = 1,
                                   pady = 1, fg = "white", bg = "#00006F", command = aumentar_dados)
    btn_aumentar_dados.place(relwidth = 0.27, relheight = 0.1, relx = 0.46, rely = 0.89)

    # Criar botao resonsavel por aumentar a quantidade de dados
    btn_classificadores = Tk.Button(root, text = "Classificadores", padx = 1,
                                    pady = 1, fg = "white", bg = "#00006F", command = construir_menu_classificador)
    btn_classificadores.place(relwidth = 0.24, relheight = 0.1, relx = 0.74, rely = 0.89)

    info = '\nTrabalho de Processamento e Análise da Imagem'
    info += '\n\n Descrição: Essa aplicação tem como intuito analisar imagens\n de raio X de joelho e classificar os tipos de osteoartrite.'

    info += '\n\n * Cortar Imagem: Selecionar uma imagem e fazer um corte\n manual para ajudar na identificação da junta do joelho.'
    info += '\n\n * Comparar: Escolha uma imagem que represente uma\n subregião de outra imagem e encontraremos\n essa região em uma segunda imagem.'
    info += '\n\n * Aumentar dados: Selecione um diretório contendo\n imagens e serão gerados dois diretórios com\n as imagens originais modificadas.'
    info += '\n\n * Classificadores: Selecione um dos classificadores\n implementados para análise de imagens.'
    informacao = Tk.Label(root, text= info, bg = "#202020", fg = "white", anchor = 'n', font=("Arial", 12))
    informacao.place(relwidth = 0.98, relheight = 0.8, relx = 0.01, rely = 0.01)

"""
Nome: construir_menu_classificador
Funcao: Criar menu com configuração reponsavel
por disponibilizar os classificadores utilizados.
"""
def construir_menu_classificador():
    # Configurar tela
    limpar_menu()
    root.title("Escolha o CLassificador")
    root.minsize(330, 50)
    root.maxsize(330, 50)
    screen = Tk.Canvas(root, height = 50, width= 330, bg = "#202020")
    screen.pack()

    # Criar botao para voltar o menu principal
    btn_shufflenet = Tk.Button(root, text = "<", padx = 1, pady = 1,
                               fg = "white", bg = "#00006F", command = construir_menu_principal)
    btn_shufflenet.place(relwidth = 0.10, relheight = 0.8, relx = 0.02, rely = 0.1)

    # Criar botao para disponibilizar o menu do shufflenet
    btn_shufflenet = Tk.Button(root, text = "Shufflenet", padx = 1, pady = 1,
                               fg = "white", bg = "#00006F", command = construir_menu_shufflenet)
    btn_shufflenet.place(relwidth = 0.31, relheight = 0.8, relx = 0.13, rely = 0.1)

    # Criar botao para disponibilizar o menu do svm
    btn_shufflenet = Tk.Button(root, text = "SVM", padx = 1, pady = 1,
                               fg = "white", bg = "#00006F", command = construir_menu_svm)
    btn_shufflenet.place(relwidth = 0.23, relheight = 0.8, relx = 0.45, rely = 0.1)

    # Criar botao para disponibilizar o menu do xgboost
    btn_shufflenet = Tk.Button(root, text = "XGBoost", padx = 1, pady = 1,
                               fg = "white", bg = "#00006F", command = construir_menu_xgboost)
    btn_shufflenet.place(relwidth = 0.29, relheight = 0.8, relx = 0.69, rely = 0.1)

"""
Nome: construir_menu_principal
Funcao: Criar menu com configuração reponsavel
por salvar a imagem obtida pelo corte e a imagem
resultante da comparacao.
"""
def construir_menu_salvar_imagem():
    # Configurar tela
    limpar_menu()
    root.title("Aviso")
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
    root.title("Escolha a Opcao - ShuffleNet")
    root.minsize(380, 50)
    root.maxsize(380, 50)
    screen = Tk.Canvas(root, height = 50, width= 380, bg = "#202020")
    screen.pack()

    # Criar botao para voltar o menu principal
    btn_shufflenet = Tk.Button(root, text = "<", padx = 1, pady = 1,
                               fg = "white", bg = "#00006F", command = construir_menu_classificador)
    btn_shufflenet.place(relwidth = 0.10, relheight = 0.8, relx = 0.02, rely = 0.1)

    # Treina o classificador
    btn_shufflenet = Tk.Button(root, text = "Treinar", padx = 1, pady = 1,
                               fg = "white", bg = "RED", command = shufflenet)
    btn_shufflenet.place(relwidth = 0.2, relheight = 0.8, relx = 0.13, rely = 0.1)

    # Apresenta os resultados dos testes do treinamento
    btn_shufflenet = Tk.Button(root, text = "Apresentar Resultados", padx = 1, pady = 1,
                               fg = "white", bg = "#00006F", command = apresentar_resultados_shufflenet)
    btn_shufflenet.place(relwidth = 0.4, relheight = 0.8, relx = 0.34, rely = 0.1)

    # Possibilita classificar uma imagem selecionada
    btn_shufflenet = Tk.Button(root, text = "Classificar", padx = 1, pady = 1,
                               fg = "white", bg = "#00006F", command = classificar_shufflenet)
    btn_shufflenet.place(relwidth = 0.23, relheight = 0.8, relx = 0.75, rely = 0.1)

"""
Nome: construir_menu_svm
Funcao: Criar menu com opcoes do classsificador.
"""
def construir_menu_svm():
    # Configurar tela
    limpar_menu()
    root.title("Escolha a Opcao - SVM")
    root.minsize(380, 50)
    root.maxsize(380, 50)
    screen = Tk.Canvas(root, height = 50, width= 380, bg = "#202020")
    screen.pack()

    # Criar botao para voltar o menu principal
    btn_shufflenet = Tk.Button(root, text = "<", padx = 1, pady = 1,
                               fg = "white", bg = "#00006F", command = construir_menu_classificador)
    btn_shufflenet.place(relwidth = 0.10, relheight = 0.8, relx = 0.02, rely = 0.1)

    # Treina o classificador
    btn_shufflenet = Tk.Button(root, text = "Treinar", padx = 1, pady = 1,
                               fg = "white", bg = "RED", command = SVM)
    btn_shufflenet.place(relwidth = 0.2, relheight = 0.8, relx = 0.13, rely = 0.1)

    # Apresenta os resultados dos testes do treinamento
    btn_shufflenet = Tk.Button(root, text = "Apresentar Resultados", padx = 1, pady = 1,
                               fg = "white", bg = "#00006F", command = construir_menu_svm_apresentar)
    btn_shufflenet.place(relwidth = 0.4, relheight = 0.8, relx = 0.34, rely = 0.1)

    # Possibilita classificar uma imagem selecionada
    btn_shufflenet = Tk.Button(root, text = "Classificar", padx = 1, pady = 1,
                               fg = "white", bg = "#00006F", command = construir_menu_svm_classificar)
    btn_shufflenet.place(relwidth = 0.23, relheight = 0.8, relx = 0.75, rely = 0.1)

"""
Nome: construir_menu_xgboost
Funcao: Criar menu com opcoes do classsificador.
"""
def construir_menu_svm_apresentar():
    # Configurar tela
    limpar_menu()
    root.title("Dados - SMV")
    root.minsize(300, 50)
    root.maxsize(300, 50)
    screen = Tk.Canvas(root, height = 50, width= 300, bg = "#202020")
    screen.pack()

    # Criar botao com opcao para salvar
    btn_salvar = Tk.Button(root, text = "<", padx = 1, pady = 1,
                           fg = "white", bg = "#00006F", command = construir_menu_svm)
    btn_salvar.place(relwidth = 0.2, relheight = 0.8, relx = 0.02, rely = 0.1)

    # Criar botao com opcao para salvar
    btn_salvar = Tk.Button(root, text = "Multiclasse", padx = 1, pady = 1,
                           fg = "white", bg = "#00006F", command = apresentar_resultados_svm)
    btn_salvar.place(relwidth = 0.36, relheight = 0.8, relx = 0.23, rely = 0.1)

    # Criar botao com opcao de nao salvar
    btn_nao_salvar = Tk.Button(root, text = "Binário", padx = 1, pady = 1,
                               fg = "white", bg = "#00006F", command = apresentar_resultados_svmB)
    btn_nao_salvar.place(relwidth = 0.36, relheight = 0.8, relx = 0.62, rely = 0.1)

"""
Nome: construir_menu_xgboost
Funcao: Criar menu com opcoes do classsificador.
"""
def construir_menu_svm_classificar():
    # Configurar tela
    limpar_menu()
    root.title("Classificar - SMV")
    root.minsize(300, 50)
    root.maxsize(300, 50)
    screen = Tk.Canvas(root, height = 50, width= 300, bg = "#202020")
    screen.pack()

    # Criar botao com opcao para salvar
    btn_salvar = Tk.Button(root, text = "<", padx = 1, pady = 1,
                           fg = "white", bg = "#00006F", command = construir_menu_svm)
    btn_salvar.place(relwidth = 0.2, relheight = 0.8, relx = 0.02, rely = 0.1)

    # Criar botao com opcao para salvar
    btn_salvar = Tk.Button(root, text = "Multiclasse", padx = 1, pady = 1,
                           fg = "white", bg = "#00006F", command = classificar_svm)
    btn_salvar.place(relwidth = 0.36, relheight = 0.8, relx = 0.23, rely = 0.1)

    # Criar botao com opcao de nao salvar
    btn_nao_salvar = Tk.Button(root, text = "Binário", padx = 1, pady = 1,
                               fg = "white", bg = "#00006F", command = classificar_svmB)
    btn_nao_salvar.place(relwidth = 0.36, relheight = 0.8, relx = 0.62, rely = 0.1)

"""
Nome: construir_menu_xgboost
Funcao: Criar menu com opcoes do classsificador.
"""
def construir_menu_xgboost():
    # Configurar tela
    limpar_menu()
    root.title("Escolha a Opcao - XGBoost")
    root.minsize(380, 50)
    root.maxsize(380, 50)
    screen = Tk.Canvas(root, height = 50, width= 380, bg = "#202020")
    screen.pack()

    # Criar botao para voltar o menu principal
    btn_shufflenet = Tk.Button(root, text = "<", padx = 1, pady = 1,
                               fg = "white", bg = "#00006F", command = construir_menu_classificador)
    btn_shufflenet.place(relwidth = 0.10, relheight = 0.8, relx = 0.02, rely = 0.1)

    # Treina o classificador
    btn_shufflenet = Tk.Button(root, text = "Treinar", padx = 1, pady = 1,
                               fg = "white", bg = "RED", command = XGBoost)
    btn_shufflenet.place(relwidth = 0.2, relheight = 0.8, relx = 0.13, rely = 0.1)

    # Apresenta os resultados dos testes do treinamento
    btn_shufflenet = Tk.Button(root, text = "Apresentar Resultados", padx = 1, pady = 1,
                               fg = "white", bg = "#00006F", command = construir_menu_xgboost_apresentar)
    btn_shufflenet.place(relwidth = 0.4, relheight = 0.8, relx = 0.34, rely = 0.1)

    # Possibilita classificar uma imagem selecionada
    btn_shufflenet = Tk.Button(root, text = "Classificar", padx = 1, pady = 1,
                               fg = "white", bg = "#00006F", command = construir_menu_xgboost_classificar)
    btn_shufflenet.place(relwidth = 0.23, relheight = 0.8, relx = 0.75, rely = 0.1)

"""
Nome: construir_menu_xgboost
Funcao: Criar menu com opcoes do classsificador.
"""
def construir_menu_xgboost_apresentar():
    # Configurar tela
    limpar_menu()
    root.title("Dados - XGBoost")
    root.minsize(300, 50)
    root.maxsize(300, 50)
    screen = Tk.Canvas(root, height = 50, width= 300, bg = "#202020")
    screen.pack()

    # Criar botao com opcao para salvar
    btn_salvar = Tk.Button(root, text = "<", padx = 1, pady = 1,
                           fg = "white", bg = "#00006F", command = construir_menu_xgboost)
    btn_salvar.place(relwidth = 0.2, relheight = 0.8, relx = 0.02, rely = 0.1)

    # Criar botao com opcao para salvar
    btn_salvar = Tk.Button(root, text = "Multiclasse", padx = 1, pady = 1,
                           fg = "white", bg = "#00006F", command = apresentar_resultados_xgboost)
    btn_salvar.place(relwidth = 0.36, relheight = 0.8, relx = 0.23, rely = 0.1)

    # Criar botao com opcao de nao salvar
    btn_nao_salvar = Tk.Button(root, text = "Binário", padx = 1, pady = 1,
                               fg = "white", bg = "#00006F", command = apresentar_resultados_xgboostB)
    btn_nao_salvar.place(relwidth = 0.36, relheight = 0.8, relx = 0.62, rely = 0.1)

"""
Nome: construir_menu_xgboost
Funcao: Criar menu com opcoes do classsificador.
"""
def construir_menu_xgboost_classificar():
    # Configurar tela
    limpar_menu()
    root.title("Classificar - XGBoost")
    root.minsize(300, 50)
    root.maxsize(300, 50)
    screen = Tk.Canvas(root, height = 50, width= 300, bg = "#202020")
    screen.pack()

    # Criar botao com opcao para salvar
    btn_salvar = Tk.Button(root, text = "<", padx = 1, pady = 1,
                           fg = "white", bg = "#00006F", command = construir_menu_xgboost)
    btn_salvar.place(relwidth = 0.2, relheight = 0.8, relx = 0.02, rely = 0.1)

    # Criar botao com opcao para salvar
    btn_salvar = Tk.Button(root, text = "Multiclasse", padx = 1, pady = 1,
                           fg = "white", bg = "#00006F", command = classificar_xgboost)
    btn_salvar.place(relwidth = 0.36, relheight = 0.8, relx = 0.23, rely = 0.1)

    # Criar botao com opcao de nao salvar
    btn_nao_salvar = Tk.Button(root, text = "Binário", padx = 1, pady = 1,
                               fg = "white", bg = "#00006F", command = classificar_xgboostB)
    btn_nao_salvar.place(relwidth = 0.36, relheight = 0.8, relx = 0.62, rely = 0.1)

"""
Nome: construir_menu_shufflenet
Funcao: Criar menu com opcoes do classsificador.
"""
def apresentar_dados(info):
    # Configurar tela
    limpar_menu()
    root.title("Informações")
    root.minsize(380, 500)
    root.maxsize(380, 500)
    screen = Tk.Canvas(root, height = 500, width= 380, bg = "#202020")
    screen.pack()

    accuracy = Tk.Label(root, text= str(info), bg = "#202020", fg = "white")
    accuracy.place(relwidth = 0.98, relheight = 0.88, relx = 0.01, rely = 0.01)

    # Criar botao para voltar o menu principal
    btn_shufflenet = Tk.Button(root, text = "Voltar", padx = 1, pady = 1,
                               fg = "white", bg = "#00006F", command = construir_menu_classificador)
    btn_shufflenet.place(relwidth = 0.98, relheight = 0.10, relx = 0.01, rely = 0.89)

def isNaN(num):
    return num!= num

construir_menu_principal() # Chamar primeira criacao/configuracao de tela
root.mainloop() # Deixar tela aberta
