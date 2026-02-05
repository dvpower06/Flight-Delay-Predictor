import random
import math
import matplotlib.pyplot as plt

alpha = 0.2 #velocidade de aprendizagem

#------------------CÓDIGO GENÉRICO PARA CRIAR, TREINAR E USAR UMA REDE COM UMA CAMADA ESCONDIDA------------
def make(nx, nz, ny):
    """Funcao que cria, inicializa e devolve uma rede neuronal, incluindo
    a criacao das diversas listas, bem como a inicializacao das listas de pesos. 
    Note-se que sao incluidas duas unidades extra, uma de entrada e outra escondida, 
    mais os respectivos pesos, para lidar com os tresholds; note-se tambem que, 
    tal como foi discutido na teorica, as saidas destas unidades estao sempre a -1.
    por exemplo, a chamada make(3, 5, 2) cria e devolve uma rede 3x5x2"""
    #a rede neuronal é num dicionario com as seguintes chaves:
    # nx     numero de entradas
    # nz     numero de unidades escondidas
    # ny     numero de saidas
    # x      lista de armazenamento dos valores de entrada
    # z      array de armazenamento dos valores de activacao das unidades escondidas
    # y      array de armazenamento dos valores de activacao das saidas
    # wzx    array de pesos entre a camada de entrada e a camada escondida
    # wyz    array de pesos entre a camada escondida e a camada de saida
    # dz     array de erros das unidades escondidas
    # dy     array de erros das unidades de saida    
    
    nn = {'nx':nx, 'nz':nz, 'ny':ny, 'x':[], 'z':[], 'y':[], 'wzx':[], 'wyz':[], 'dz':[], 'dy':[]}
    
    nn['wzx'] = [[random.uniform(-0.5,0.5) for _ in range(nn['nx'] + 1)] for _ in range(nn['nz'])]
    nn['wyz'] = [[random.uniform(-0.5,0.5) for _ in range(nn['nz'] + 1)] for _ in range(nn['ny'])]
    
    return nn

def sig(inp):
    """Funcao de activacao (sigmoide)"""
    return 1.0/(1.0 + math.exp(-inp))


def forward(nn, input):
    """Função que recebe uma rede nn e um padrao de entrada input (uma lista) 
    e faz a propagacao da informacao para a frente ate as saidas"""
    #copia a informacao do vector de entrada input para a listavector de inputs da rede nn  
    nn['x']=input.copy()
    #adiciona a entrada a -1 que vai permitir a aprendizagem dos limiares
    nn['x'].append(-1)
    #calcula a activacao da unidades escondidas
    for i in range (nn['nz']):
        nn['z']=[sig(sum([x*w for x, w in zip(nn['x'], nn['wzx'][i])])) for i in range(nn['nz'])]
        #adiciona a entrada a -1 que vai permitir a aprendizagem dos limiares
        nn['z'].append(-1)
        #calcula a activacao da unidades de saida
        nn['y']=[sig(sum([z*w for z, w in zip(nn['z'], nn['wyz'][i])])) for i in range(nn['ny'])]
 
   
def error(nn, output):
    """Funcao que recebe uma rede nn com as activacoes calculadas
       e a lista output de saidas pretendidas e calcula os erros
       na camada escondida e na camada de saida"""
    nn['dy']=[y*(1-y)*(o-y) for y,o in zip(nn['y'], output)]
    zerror=[sum([nn['wyz'][i][j]*nn['dy'][i] for i in range(nn['ny'])]) for j in range(nn['nz'])]
    nn['dz']=[z*(1-z)*e for z, e in zip(nn['z'], zerror)]
 
 
def update(nn):
    """funcao que recebe uma rede com as activacoes e erros calculados e
    actualiza as listas de pesos"""
    
    nn['wzx'] = [ [w+x*nn['dz'][i]*alpha for w, x in zip(nn['wzx'][i], nn['x'])] for i in range(nn['nz'])]
    nn['wyz'] = [ [w+z*nn['dy'][i]*alpha for w, z in zip(nn['wyz'][i], nn['z'])] for i in range(nn['ny'])]
    

def iterate(i, nn, input, output):
    """Funcao que realiza uma iteracao de treino para um dado padrao de entrada input
    com saida desejada output"""
    forward(nn, input)
    error(nn, output)
    update(nn)
    print('%03i: %s -----> %s : %s' %(i, input, output, nn['y']))
    

#-----------------------CÓDIGO QUE PERMITE CRIAR E TREINAR REDES PARA APRENDER AS FUNÇÕES BOOLENAS--------------------
"""Funcao que cria uma rede 2x2x1 e treina a função lógica AND
A função recebe como entrada o número de épocas com que se pretende treinar a rede"""
def train_and(epocas):
    net = make(2, 2, 1)
    for i in range(epocas):
        iterate(i, net, [0, 0], [0])
        iterate(i, net, [0, 1], [0])
        iterate(i, net, [1, 0], [0])
        iterate(i, net, [1, 1], [1])
    return net
    
"""Funcao que cria uma rede 2x2x1 e treina um OR
A função recebe como entrada o número de épocas com que se pretende treinar a rede"""
def train_or(epocas):
    net = make(2, 2, 1)
    for i in range(epocas):
        iterate(i, net, [0, 0], [0])
        iterate(i, net, [0, 1], [1])
        iterate(i, net, [1, 0], [1])
        iterate(i, net, [1, 1], [1]) 
    return net

"""Funcao que cria uma rede 2x2x1 e treina um XOR
A função recebe como entrada o número de épocas com que se pretende treinar a rede"""
def train_xor(epocas):
    net = make(2, 2, 1)
    for i in range(epocas):
        iterate(i, net, [0, 0], [0])
        iterate(i, net, [0, 1], [1])
        iterate(i, net, [1, 0], [1])
        iterate(i, net, [1, 1], [0]) 
    return net


#-------------------------CÓDIGO QUE IRÁ PERMITIR CRIAR UMA REDE PARA APRENDER A CLASSIFICAR VOOS DE AVIÃO---------    

"""Funcao principal do nosso programa para prever se um voo chegará com ou sem atraso:
cria os conjuntos de treino e teste, chama a funcao que cria e treina a rede e, por fim, 
a funcao que a testa. A funcao recebe como argumento o ficheiro correspondente ao dataset 
que deve ser usado, os tamanhos das camadas de entrada, escondida e saída,
o numero de epocas que deve ser considerado no treino e os tamanhos dos conjuntos de treino e 
teste"""
def run_flights(file, input_size, hidden_size, output_size, epochs, training_set_size, test_set_size):
    #A definir pelos estudantes
    training_set, test_set = build_sets(file,training_set_size, test_set_size)

    net = train_flights(input_size, hidden_size, output_size,training_set, test_set, epochs)

    test_flights(net, test_set, printing=True)


"""Funcao que cria os conjuntos de treino e de teste a partir dos dados
armazenados em f ('DataSet1.csv'). A funcao le cada linha, 
tranforma-a numa lista de valores e chama a funcao translate para a colocar no 
formato adequado para o padrao de treino. Estes padroes são colocados numa lista.
A função recebe como argumentos o nº de exemplos que devem ser considerados no conjunto de 
treino --->x e o nº de exemplos que devem ser considerados no conjunto de teste ------> y
Finalmente, devolve duas listas, uma com x padroes (conjunto de treino)
e a segunda com y padrões (conjunto de teste). Atenção que x+y não pode ultrapassar o nº 
de estudantes disponível no dataset"""       
def build_sets(nome_f, x, y):

    padroes_treino = []

    file = open(nome_f, 'r')
    linhas = file.readlines()
    file.close()

    linhas_dados = linhas[1:]

    for linha in linhas_dados:
        linha = linha.strip()
        dados = linha.split(',')
        padrao = translate(dados)
        padroes_treino.append(padrao)

    return padroes_treino[:x], padroes_treino[x:x+y]

    #A definir pelos estudantes


"""A função translate recebe cada lista de valores que caracterizam um vooda
e transforma-a num padrão de treino. Cada padrão é uma lista com o seguinte formato 
[padrao_de_entrada, classe_do_voo, padrao_de_saida]
O enunciado do trabalho explica de que forma deve ser obtido o padrão de entrada
"""

LISTA_CARRIERS = ['WN', 'NK', 'F9', 'AA', 'UA', 'DL', 'MQ', 'AS', 'B6', 'HA']
LISTA_AEROPORTOS = ['DEN', 'TPA', 'ORD', 'SAN', 'JFK', 'ATL', 'SFO', 'DTW', 'MSP', 'PHX', 'DFW', 'BWI', 'EWR', 'CLT', 'MIA', 'LAX', 'SEA', 'IAH', 'BOS', 'LAS']
MIN_E_MAX = {
    'Month':(1,12),
    'DayOfWeek':(1,7),
    'CRSDepTime':(0, 1439),
    'Distance':(150, 3000)
}



def converterHHMM(hora):
    return hora/60


def translate(lista):
    #A definir pelos estudantes
    month = int(lista[0])
    DayOfWeek = int(lista[1])
    CRSDepTime = int(lista[2])
    UniqueCarrier = lista[3]
    Origin = lista[4]
    Dest = lista[5]
    Distance = int(lista[6])
    classe = int(lista[7])

    
    entrada_numeros = [normaliza_valores(month, MIN_E_MAX['Month']), normaliza_valores(DayOfWeek, MIN_E_MAX['DayOfWeek']), converterHHMM(CRSDepTime), normaliza_valores(Distance, MIN_E_MAX['Distance'])]
    entrada_carrier = converte_categ_numerico(UniqueCarrier)
    entrada_origin = converte_categ_numerico(Origin)
    entrada_dest = converte_categ_numerico(Dest)
    

    padrao_de_entrada = entrada_numeros + entrada_carrier + entrada_origin + entrada_dest
      
    if classe == 0:
        padrao_de_saida = [1,0]
    elif classe == 1:
        padrao_de_saida = [0,1]
    
    padrao = [padrao_de_entrada, classe, padrao_de_saida]

    return padrao

#Função que converte valores categóricos para a codificação onehot                
def converte_categ_numerico(instancia):
    #A definir pelos estudantes

    if instancia in LISTA_CARRIERS:
        descricao = LISTA_CARRIERS
    elif instancia in LISTA_AEROPORTOS:
        descricao = LISTA_AEROPORTOS
    else:
        return [0] * len(LISTA_CARRIERS)  

    binarios = [0]* len(descricao)

    idx_instancia = descricao.index(instancia)
    binarios[idx_instancia] = 1

    return binarios


"""Função que normaliza os valores necessários"""   
def normaliza_valores(valor, lista):
    #A definir pelos estudantes
    max_value = max(lista)
    min_value = min(lista)

    return (valor - min_value) / (max_value - min_value)


"""Cria a rede e chama a funçao iterate para a treinar. A função recebe como argumento 
o conjunto de treino, os tamanhos das camadas de entrada, escondida e saída e o número 
de épocas que irão ser usadas para fazer o treino"""
def train_flights(input_size, hidden_size, output_size, training_set, test_set, epochs):
    #A definir pelos estudantes

    exatidao_test_set = []
    precisao_test_set = []
    cobertura_test_set = []
    f1_score_test_set = []

    exatidao_training_set = []
    precisao_training_set = []
    cobertura_training_set = []
    f1_score_training_set = []

    net = make(input_size, hidden_size, output_size)

    for i in range(epochs):
        for padrao in training_set:
            entrada = padrao[0]
            saida = padrao[2]
            iterate(i, net, entrada, saida )

        resultado_test =  test_flights(net, test_set, printing=False)
        exatidao_test_set.append(resultado_test['exatidao'])
        precisao_test_set.append(resultado_test['precisao'])
        cobertura_test_set.append(resultado_test['cobertura'])
        f1_score_test_set.append(resultado_test['f1-score'])

        resultado_train = test_flights(net, training_set, printing=False)
        exatidao_training_set.append(resultado_train['exatidao'])
        precisao_training_set.append(resultado_train['precisao'])
        cobertura_training_set.append(resultado_train['cobertura'])
        f1_score_training_set.append(resultado_train['f1-score'])



    epocas = list(range(1, epochs + 1))

    fig, axs = plt.subplots(2, 2)

    axs[0, 0].plot(epocas, exatidao_training_set, label="Treino")
    axs[0, 0].plot(epocas, exatidao_test_set, label="Teste")
    axs[0, 0].set_title("Exatidao")
    axs[0, 0].set_xlabel("Epocas")
    axs[0, 0].set_ylabel("Exatidao (%)")
    axs[0, 0].legend()

    axs[0, 1].plot(epocas, precisao_training_set, label="Treino")
    axs[0, 1].plot(epocas, precisao_test_set, label="Teste")
    axs[0, 1].set_title("Precisao")
    axs[0, 1].set_xlabel("Epocas")
    axs[0, 1].set_ylabel("Precisao")
    axs[0, 1].legend()

    axs[1, 0].plot(epocas, cobertura_training_set, label="Treino")
    axs[1, 0].plot(epocas, cobertura_test_set, label="Teste")
    axs[1, 0].set_title("Cobertura")
    axs[1, 0].set_xlabel("Epocas")
    axs[1, 0].set_ylabel("Cobertura")
    axs[1, 0].legend()

    axs[1, 1].plot(epocas, f1_score_training_set, label="Treino")
    axs[1, 1].plot(epocas, f1_score_test_set, label="Teste")
    axs[1, 1].set_title("F1-score")
    axs[1, 1].set_xlabel("Epocas")
    axs[1, 1].set_ylabel("F1-score")
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()

    return net



"""Funcao que avalia a precisao da rede treinada, utilizando o conjunto de teste ou treino.
Para cada padrao do conjunto chama a funcao forward e determina a classe do voo
que corresponde ao maior valor da lista de saida. A classe determinada pela rede
deve ser comparada com a classe real,sendo contabilizado o número de respostas corretas. 
A função calcula a percentagem de respostas corretas, 
o nº de VP,FP,VN, FN, precisão, cobertura e f1-score""" 
def test_flights(net, test_set, printing = True):
    #A definir pelos estudantes
    total_respostas = 0
    corretas=0
    n_voo = 0
    frase = ''
    fp=0
    tn=0
    tp=0
    fn=0

    for padrao in test_set:
        total_respostas+=1
        n_voo+=1
        entradas = padrao[0]
        real = padrao[1]

        forward(net,entradas)
        previsto = retranslate(net['y'])

        if previsto == real:
            corretas+=1
            if previsto==0:
                tn+=1
                frase = 'não está atrasado e na de facto não está atrasado.'
            elif previsto == 1:
                tp+=1
                frase = ' está atrasado e de facto ele está atrasado.'

        else:
             if previsto==0:
                fn+=1
                frase = 'está atrasado e na realidade ele não está atrasado.'
             elif previsto == 1:
                fp+=1
                frase = 'não está atrasado e na realidade ele está atrasado.'

        if printing:
            print('A rede prevê que o voo nº' + str(n_voo) + frase)


    if (tp + fp) > 0:
        precisao = tp / (tp + fp)
    else:
        precisao = 0
    
    if (tp + fn) > 0:
        cobertura = tp / (tp + fn)
    else:
        cobertura = 0
    
    if (precisao + cobertura) > 0:
        f1_score = 2*((precisao*cobertura)/(precisao+cobertura))
    else:
        f1_score = 0
    
    percentagem_corretas = corretas*100/total_respostas

    if printing:
        print('Success_rate: ' + str(percentagem_corretas))
    else:
        dic = {'exatidao': percentagem_corretas,
            'matriz': [tp, fp, tn, fn] ,
            'precisao': precisao, 
            'cobertura': cobertura, 
            'f1-score': f1_score 
            }
        
        return dic


  
"""Recebe o padrao de saida da rede e devolve a situação de atraso do voo.
A situação de atraso corresponde ao indice da saida com maior valor."""  
def retranslate(out):
    return out.index(max(out))



"""
if __name__ == "__main__":
    #Vamos treinar durante 1000 épocas uma rede para aprender a função logica AND
    #Faz testes para números de épocas diferentes e para as restantes funções lógicas já implementadas
    rede_or = train_or(1000)
    #Agora vamos ver se ela aprendeu bem
    tabela_verdade = {(0,0): 0, (0,1): 1, (1,0): 1, (1,1): 1}
    for linha in tabela_verdade:
        forward(rede_or, list(linha))
        print('A rede determinou %s para a entrada %d OR %d quando devia ser %d'
              %(rede_or['y'], linha[0], linha[1], tabela_verdade[linha]))
        
"""

if __name__ == "__main__":
    input_size, hidden_size, output_size = 18, 5, 2
    epochs = 5
    training_set_size = 800
    test_set_size = 200
    file = 'DataSet1.csv'
    run_flights(file, input_size, hidden_size, output_size, epochs, training_set_size, test_set_size)