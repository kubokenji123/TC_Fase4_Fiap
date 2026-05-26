# TC_Fase4_Fiap

Aplicação Streamlit para previsão de nível de obesidade usando um modelo Random Forest treinado.

**Conteúdo do repositório**
- Files/: código fonte, modelo treinado e dados.
	- Files/app.py — app Streamlit.
	- Files/random_forest_model.pkl — modelo treinado (joblib).
	- Files/scaler.pkl — scaler usado no pré-processamento.
	- Files/label_encoder.pkl — encoder de rótulos.
	- Files/Obesity.csv — dataset original.
- requirements.txt — dependências do Python.

**Como executar localmente**
1. Criar e ativar um ambiente virtual (recomendado):
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
```
2. Instalar dependências:
```bash
pip install -r Files/requirements.txt
```
3. Rodar a aplicação Streamlit:
```bash
streamlit run Files/app.py
```

**Observações importantes**
- O app carrega os arquivos `.pkl` a partir do diretório `Files/`, portanto mantenha os artefatos (`random_forest_model.pkl`, `scaler.pkl`, `label_encoder.pkl`) nesse diretório.
- Se receber erro ao carregar os arquivos, verifique permissões e caminhos relativos.

**Link público (deploy)**
- https://bsdmwjceuphljmkmpgvddn.streamlit.app/

---
Atualizado para incluir instruções de execução local e estrutura do projeto.