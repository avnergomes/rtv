\# 📊 Dashboard RTVs – Paraná



Este repositório contém um painel interativo desenvolvido em \*\*Streamlit\*\* para monitoramento dos Relatórios Técnicos de Vistoria (RTVs) no Paraná.



\## 🚀 Funcionalidades

\- KPIs (total de RTVs, extensão total, % entregues, municípios atendidos)

\- Gráfico incremental de entregas (Curva S)

\- Distribuição por Status e Pavimento

\- Comparativo por Região

\- Mapa interativo com intensidade de cor (Extensão ou Quantidade de RTVs)



\## 🌍 Deploy

O app está publicado em:  

👉 \[Abrir no Streamlit Cloud](https://dashboard-rtvs.streamlit.app)



\## 🛠 Como rodar localmente

```bash

git clone https://github.com/SEU\_USUARIO/dashboard-rtvs.git

cd dashboard-rtvs

pip install -r requirements.txt

streamlit run app.py





---



\## 🚀 Deploy no Streamlit Cloud

1\. Vá em \[Streamlit Cloud](https://share.streamlit.io/).  

2\. Clique em \*\*New App\*\*.  

3\. Conecte sua conta ao \*\*GitHub\*\*.  

4\. Escolha o repositório (`dashboard-rtvs`) e a branch principal (`main` ou `master`).  

5\. No campo \*\*Main file path\*\*, coloque `app.py`.  

6\. Clique em \*\*Deploy\*\*.  



Ele vai:

\- Instalar dependências do `requirements.txt`.  

\- Rodar `app.py`.  

\- Gerar uma URL pública do painel.  



---



\## 🔑 Observações importantes

\- O \*\*Google Sheets\*\* precisa estar \*\*compartilhado como público com link\*\* (já está no seu caso, pois o link `export?format=xlsx` funciona sem login).  

\- O arquivo `mun\_PR.json` deve estar no mesmo nível do `app.py` no repositório.  

\- Se o Streamlit Cloud travar por conta do `geopandas`, posso te passar uma versão que carrega o \*\*GeoJSON direto com `json` + `pydeck`\*\* sem depender de `geopandas`.  



---



👉 Você quer que eu já \*\*adapte o `app.py` para não depender do `geopandas`\*\* (só `json` + `pandas` + `pydeck`) e evitar qualquer dor de cabeça no deploy?





