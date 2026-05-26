import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Carregar o modelo, o scaler e o label encoder usando caminhos relativos
BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(str(BASE_DIR / 'random_forest_model.pkl'))
scaler = joblib.load(str(BASE_DIR / 'scaler.pkl'))
label_encoder = joblib.load(str(BASE_DIR / 'label_encoder.pkl'))

# Título da aplicação
st.title('Previsão de Nível de Obesidade')
st.write('Este aplicativo prevê o nível de obesidade de uma pessoa com base em suas características.')

# Coleta de Inputs do Usuário
st.header('Informações do Usuário')

# Inputs numéricos
age = st.slider('Idade', 10, 80, 25)
height = st.slider('Altura (m)', 1.0, 2.0, 1.70, 0.01)
weight = st.slider('Peso (kg)', 30.0, 180.0, 70.0, 0.1)
fcvc = st.slider('Frequência de consumo de vegetais (1-3)', 1.0, 3.0, 2.0, 0.1)
ncp = st.slider('Número de refeições principais (1-4)', 1.0, 4.0, 3.0, 0.1)
ch2o = st.slider('Consumo de água (litros/dia)', 1.0, 3.0, 2.0, 0.1)
faf = st.slider('Frequência de atividade física (0-3)', 0.0, 3.0, 1.0, 0.1)
tue = st.slider('Tempo de uso de dispositivos tecnológicos (0-2)', 0.0, 2.0, 1.0, 0.1)

# Inputs categóricos
gender = st.selectbox('Gênero', ['Female', 'Male'])
family_history = st.selectbox('Histórico familiar de obesidade', ['yes', 'no'])
favc = st.selectbox('Consome alimentos altamente calóricos com frequência?', ['yes', 'no'])
caec = st.selectbox('Come entre as refeições?', ['no', 'Sometimes', 'Frequently', 'Always'])
smoke = st.selectbox('Você fuma?', ['yes', 'no'])
scc = st.selectbox('Você monitora as calorias?', ['yes', 'no'])
calc = st.selectbox('Frequência de consumo de álcool', ['no', 'Sometimes', 'Frequently', 'Always'])
mtrans = st.selectbox('Meio de transporte principal', ['Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike'])

# Mapear inputs para o formato do DataFrame original (pré One-Hot Encoding)
input_data = {
    'Gender': gender,
    'Age': age,
    'Height': height,
    'Weight': weight,
    'family_history': family_history,
    'FAVC': favc,
    'FCVC': fcvc,
    'NCP': ncp,
    'CAEC': caec,
    'SMOKE': smoke,
    'CH2O': ch2o,
    'SCC': scc,
    'FAF': faf,
    'TUE': tue,
    'CALC': calc,
    'MTRANS': mtrans
}

# Criar DataFrame com os inputs do usuário
input_df = pd.DataFrame([input_data])

# Precisamos garantir que as colunas categóricas no input_df tenham as mesmas colunas one-hot encoded
# que o DataFrame de treinamento original. Para isso, criaremos um DataFrame de "template"
# com todas as colunas esperadas após o one-hot encoding.

# Lista de colunas categóricas usadas para one-hot encoding no treinamento (excluindo 'Obesity')
categorical_cols = ['Gender', 'family_history', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

# O df_final possui todas as colunas que o modelo espera
# Vamos pegar as colunas de X_train_scaled, que são as colunas que o modelo espera, menos a coluna alvo
model_features = model.feature_names_in_

# Aplicar One-Hot Encoding no input_df
input_df_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

# Reindexar as colunas para corresponder às colunas de treinamento
# Adicionar colunas que podem estar faltando (com valor 0) e remover as extras
final_input_df = pd.DataFrame(columns=model_features)
final_input_df = pd.concat([final_input_df, input_df_encoded], ignore_index=True)
final_input_df = final_input_df.fillna(0) # Preencher NaN com 0 para colunas one-hot ausentes

# Garantir que todas as colunas bool False/True sejam convertidas para int 0/1
for col in final_input_df.columns:
    if final_input_df[col].dtype == 'bool':
        final_input_df[col] = final_input_df[col].astype(int)

# Ordenar colunas para garantir que a ordem seja a mesma do treinamento
final_input_df = final_input_df[model_features]

# Identificar colunas numéricas para escalar (as mesmas usadas no treinamento)
numeric_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE'] # Definido previamente no notebook

# Aplicar o scaler
final_input_df[numeric_cols] = scaler.transform(final_input_df[numeric_cols])

# Botão para fazer a previsão
if st.button('Prever Nível de Obesidade'):
    prediction_encoded = model.predict(final_input_df)
    prediction_label = label_encoder.inverse_transform(prediction_encoded)

    st.success(f'O nível de obesidade previsto é: **{prediction_label[0]}**')

st.markdown("""
### Dicionário de Dados:
*   **Gender**: Gênero.
*   **Age**: Idade.
*   **Height**: Altura em metros.
*   **Weight**: Peso em kgs.
*   **family_history**: Algum membro da família sofreu ou sofre de excesso de peso?
*   **FAVC**: Você come alimentos altamente calóricos com frequência?
*   **FCVC**: Você costuma comer vegetais nas suas refeições? (1-3)
*   **NCP**: Quantas refeições principais você faz diariamente? (1-4)
*   **CAEC**: Você come alguma coisa entre as refeições?
*   **SMOKE**: Você fuma?
*   **CH2O**: Quanta água você bebe diariamente? (1-3)
*   **SCC**: Você monitora as calorias que ingere diariamente?
*   **FAF**: Com que frequência você pratica atividade física? (0-3)
*   **TUE**: Quanto tempo você usa dispositivos tecnológicos? (0-2)
*   **CALC**: Com que frequência você bebe álcool?
*   **MTRANS**: Qual meio de transporte você costuma usar?
""")
