# 🌍 Análise Global de Consumo de Energia com Machine Learning

Bem-vindo(a) ao projeto de análise de dados energéticos!  
Neste estudo, utilizamos **técnicas de regressão e validação de modelos de machine learning** para extrair insights a partir de um conjunto de dados global de consumo energético. A proposta é entender como energia, economia e emissões de carbono se relacionam ao longo do tempo em diferentes países. ⚡📊

---

## 🛠️ Ferramentas utilizadas:
- `pandas`, `numpy` (para tratamento de dados)
- `matplotlib`, `seaborn` (para visualização dos dados)
- `scikit-learn` (para modelagem preditiva e validação)
- `xgboost` (para modelagem avançada)
- `GridSearchCV` (para ajuste de hiperparâmetros)
- `Jupyter Notebook`

---

# 📘 Descrição do Notebook

Este notebook realiza a **exploração, modelagem, validação e comparação de desempenho** de modelos preditivos com foco em dados energéticos globais.  
Ao longo do projeto, investigamos fatores como **emissões de CO₂**, **uso de energia per capita**, **PIB** e **tipos de fonte energética**.

---

## 🧩 1. Apresentação da Base e Contexto

Este estudo utiliza o **Global Energy Consumption Dataset**, uma base que traz informações detalhadas sobre o consumo energético de diversos países ao longo de mais de 20 anos.

A relevância está em compreender **como os padrões de consumo energético estão associados a fatores econômicos, ambientais e estruturais**, em um momento em que eficiência energética e sustentabilidade são prioridades globais.

---

## 📊 2. Objetivo da Pesquisa e Problema Analisado

O objetivo principal é entender como diferentes variáveis socioeconômicas e ambientais impactam o consumo de energia dos países. A partir desse entendimento, buscamos criar modelos que expliquem e possam prever o comportamento do consumo energético, auxiliando na formulação de estratégias mais eficientes e sustentáveis.


## 👥 3. Formação da Equipe

- **Matheus Benjamin Aquino Silva**  
  Projeto desenvolvido individualmente.

---

## 4. Leitura da Base de Dados no Python 🐼

Para dar início à análise, foi realizada a leitura da base de dados utilizando a biblioteca `pandas`. A base utilizada está localizada na pasta `Dados` com o nome `global_energy_consumption_pt.csv`.

---

## 5. Visualização e Entendimento Inicial da Base de Dados 👀

Antes de avançar, exploramos o dataset para entender sua estrutura e qualidade.  
- Visualizamos as primeiras linhas para conferir os dados carregados.  
- Analisamos tipos e quantidade de dados não nulos para garantir consistência.  
- Checamos valores ausentes para decidir sobre tratamento.  
- Calculamos estatísticas básicas para entender a distribuição e detectar possíveis outliers.

---

## 6. Análise Exploratória de Dados (EDA) 📊

Nesta fase, usamos gráficos e estatísticas para descobrir padrões e relações nos dados.  
- Visualizamos a evolução temporal do consumo global de energia.  
- Identificamos os 10 países com maior uso de energia renovável.  
- Exploramos correlações entre variáveis-chave com uma matriz de correlação.  
- Analisamos a distribuição do índice de preços de energia e detectamos outliers.  
- Usamos boxplots para entender a presença de valores extremos e definir tratamentos.

---

## 7. Transformação e Preparação dos Dados 🔧

Nesta etapa, tratamos os dados para garantir qualidade:  
- Identificamos e corrigimos valores nulos em colunas numéricas, preenchendo com a média.  
- Validamos os tipos de dados para assegurar consistência.  
- Essas ações previnem erros e garantem análises confiáveis nas próximas etapas.

---

## 8. Aplicação de PCA (Análise de Componentes Principais) 📊

Utilizamos PCA para reduzir a dimensionalidade dos dados, facilitando a visualização de padrões.  
Os dados foram padronizados com `StandardScaler` para equilibrar a influência das variáveis.  
A projeção em dois componentes principais permitiu observar a distribuição dos países no espaço reduzido.

---

## 9. Contando a História dos Dados 📖

- **Crescimento do Consumo de Energia:**  
  Estudamos a evolução do consumo total de energia ao longo dos anos para entender tendências globais.  
  **Conclusão:** Consumo cresce consistentemente, principalmente em países em desenvolvimento, por causa da industrialização e aumento da população.

- **Diferenças Regionais no Uso de Energia:**  
  Analisamos o consumo de energia per capita entre países desenvolvidos e em desenvolvimento.  
  **Conclusão:** Países ricos consomem mais energia por habitante; países em desenvolvimento crescem rápido, gerando desafios futuros.

- **Adoção de Energias Renováveis:**  
  Investigamos o percentual de uso de fontes renováveis por país para avaliar o avanço na transição energética.  
  **Conclusão:** Adoção ainda baixa na maioria, com poucos líderes; transição desigual e dependente de políticas locais.

- **Correlação entre Combustíveis Fósseis e Emissões:**  
  Calculamos a correlação entre dependência de combustíveis fósseis e emissões de carbono por país.  
  **Conclusão:** Correlação forte (0.92) indica relação direta; reforça a urgência da descarbonização.

- **Análise de Correlação por Ano com Heatmap:**  
  Agrupamos dados numéricos por ano para identificar relações entre variáveis relevantes.  
  **Conclusão:** Identificamos variáveis que acompanham o consumo energético, base para escolha de features preditivas.

- **Regressão Linear Simples com Variáveis Selecionadas:**  
  Exploramos a relação do consumo energético com energia per capita, emissões e preços por meio de regressão linear.  
  **Conclusão:** Todas mostraram correlação significativa; usadas juntas em regressão múltipla para prever consumo total.

## 🔢 10. Treinar um Modelo de Regressão Linear com Todas as Variáveis

Neste passo, aplicamos uma **regressão linear múltipla** com três variáveis explicativas — uso de energia per capita, emissões de carbono e índice de preços de energia — para prever o **consumo total de energia (TWH)**.

Agrupamos os dados por ano, treinamos o modelo e avaliamos seu desempenho com R² e RMSE. Também comparamos a performance da regressão linear com dois modelos mais complexos: **Random Forest** e **XGBoost**.

### ✅ Conclusão:
O modelo linear serviu como base inicial, mas os modelos mais robustos apresentaram desempenho superior, sugerindo relações mais complexas entre as variáveis.

---

## 11. Otimização do Modelo com Vários Splits Aleatórios
Neste bloco, aplicamos regressão linear múltipla com 3 variáveis fixas, mas desta vez testamos 100 divisões aleatórias entre treino e teste.
O objetivo é encontrar a melhor divisão possível com base no maior valor de R².
Essa abordagem ajuda a verificar a robustez do modelo e entender sua sensibilidade às amostras utilizadas.

## 12. Avaliação do Melhor Split com Tabela Comparativa
Neste bloco, avaliamos o melhor modelo obtido anteriormente comparando o consumo real e o previsto por ano.
Calculamos o erro percentual anual e geramos uma visualização clara com barras (real) e linha (previsto), permitindo uma análise direta do desempenho global do modelo.

---

## 🔍13 – Comparação de Modelos de Regressão

Avaliação de três modelos para prever o consumo anual de energia (TWH) usando variáveis econômicas e ambientais.

### ⚙️ Modelos testados
- Regressão Linear
- Random Forest
- XGBoost

### 🔁 Metodologia
- 100 splits aleatórios (train/test)
- Seleção do melhor R² por modelo
- Treinamento final com 100% dos dados
- Comparação com valores reais

### 📊 Resultados

| Modelo            | Erro (%) | R²     | RMSE   |
|-------------------|----------|--------|--------|
| Linear Regression | 2.12     | 0.839  | 54.18  |
| Random Forest     | 1.18     | 0.948  | 30.69  |
| **XGBoost**       | **0.00** | **1.000** | **1.01** |

### ✅ Conclusão
**XGBoost** foi o modelo mais preciso, com R² = 1.0 e erro praticamente nulo.

---

## 14 – Validação de Modelos com Novas Estratégias

Neste capítulo, validamos os modelos de regressão aplicados para prever consumo energético, adotando estratégias realistas para avaliar a capacidade de generalização.

## Principais passos

- **Avaliação inicial com Ridge e Lasso (divisão 70/30):**  
  Observou-se queda no desempenho conforme aumenta o conjunto de teste, indicando baixa generalização.

- **Ajuste de hiperparâmetros com GridSearchCV (Ridge e Lasso):**  
  Busca do melhor valor de `alpha` com validação cruzada, com melhora modesta e ainda baixa capacidade preditiva fora da amostra.

- **Random Forest com GridSearchCV:**  
  Ajuste de parâmetros principais resultou em desempenho moderado (R² ≈ 0.57), capturando mais da metade da variância dos dados.

- **XGBoost com GridSearchCV:**  
  Apesar do ajuste extenso, o modelo apresentou desempenho ruim em dados de teste, sugerindo overfitting ou necessidade de melhorias no pré-processamento.

## Conclusão

A validação rigorosa e o ajuste fino dos hiperparâmetros são essenciais para evitar overfitting e garantir modelos confiáveis e generalizáveis. Nesta etapa, o Random Forest foi o modelo com melhor desempenho prático.

---

# 15. Conclusão Geral

- XGBoost inicial apresentou overfitting, mostrando a importância da separação treino/teste.  
- Modelos lineares (Ridge, Lasso) tiveram desempenho limitado e baixa generalização.  
- Random Forest se saiu melhor, capturando relações não lineares, mas com R² médio ainda moderado (~0,57).  
- XGBoost, mesmo com ajuste extenso, não generalizou bem, sugerindo necessidade de melhor pré-processamento e dados.  
- Nenhum modelo atingiu desempenho satisfatório para generalização com os dados atuais. 

## Conclusão Final do Projeto

Este projeto explorou e modelou o consumo energético global relacionando variáveis socioeconômicas e ambientais com o objetivo de prever o consumo total de energia.

Foram testados diversos modelos, desde regressão linear até Random Forest e XGBoost, com validação rigorosa para avaliar capacidade de generalização.

Os resultados indicam que:
- Modelos simples tiveram desempenho limitado e baixa capacidade preditiva fora da amostra.
- Modelos complexos como XGBoost apresentaram overfitting, alcançando desempenho excelente nos dados de treino, porém não generalizaram bem.
- Random Forest obteve melhor equilíbrio entre ajuste e generalização, explicando aproximadamente 57% da variância nos dados de teste.

Assim, a análise realizada evidenciou as limitações dos modelos e dados utilizados, finalizando com um panorama claro dos resultados obtidos nesta etapa.

O projeto entrega uma avaliação completa da modelagem aplicada, demonstrando os desafios e as possibilidades presentes na análise preditiva do consumo energético global.


## 🧠 Pré-requisitos

Antes de rodar o projeto, você precisa ter:

✅ **Python 3.8+ instalado** no seu computador  
✅ **Jupyter Notebook** (ou **VSCode com a extensão Jupyter**)

---

## ⚙️ Como rodar o projeto

### 1. Clone este repositório

Você pode baixar o ZIP direto no botão verde “Code > Download ZIP”  
Ou usar Git:

```bash
git clone https://github.com/Matheus538Benjamin/playground-python-data.git
cd playground-python-data
```

### 2. Rode o notebook

Abra o **Jupyter Notebook** e execute o arquivo `obter_dados_api.ipynb` célula por célula.  
Você verá os dados sendo carregados, tratados e salvos automaticamente.

---

## 👤 Autor

**Matheus Benjamin Aquino Silva**  
<<<<<<< HEAD
🔗 [github.com/Matheus538Benjamin](https://github.com/Matheus538Benjamin)
=======
🔗 [github.com/Matheus538Benjamin](https://github.com/Matheus538Benjamin)
>>>>>>> 8535205 (Adiciona notebook inicial e README do Projeto Coderhouse)
