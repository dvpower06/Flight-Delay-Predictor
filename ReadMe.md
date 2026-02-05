<h1>Deep Flight Predictor✈️</h1>
Este repositório contém o desenvolvimento de um agente aprendiz baseado numa Rede Neuronal Multicamada (Multilayer Perceptron) para a previsão de atrasos em voos. <br>
 O objetivo principal é implementar um modelo de classificação binária capaz de prever se um voo sofrerá atraso (Classe 1) ou não (Classe 0) com base em atributos operacionais simulados.
<h2>Arquitetura da Rede</h2>
 - Camada de Entrada: Neurónios correspondentes ao vetor de atributos concatenado. <br>
 - Camada Escondida: 8 neurónios (configuração base).<br>
 - Camada de Saída: 2 neurónios (representando as duas classes via one-hot).<br>
<h2>Experiências Realizadas</h2>
O projeto inclui uma análise experimental detalhada variando: <br>
 - Número de neurónios na camada escondida (4 a 20).<br>
 - Número de épocas de treino (10 a 200).<br>
 - Taxa de aprendizagem (learning rate).<br>
<br>
As métricas avaliadas incluem Exatidão, Precisão, Recall (Cobertura) e F1-Score. <br>
<br>
Autor: David Power <br>
Instituição: Politécnico de Castelo Branco - Escola Superior de Tecnologia
<br>
<br>
O projeto foi desenvolvido na unidade curricular de Inteligência Artificial da Licenciatura em Engenharia Informática.
