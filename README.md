# 🛍️ E-commerce Price Prediction Dashboard

Proyecto de Cloud Document Database con Análisis Predictivo usando Azure Cosmos DB y Machine Learning.

## 🚀 Características

- ✅ Base de datos MongoDB en Azure Cosmos DB
- ✅ Recolección automática de datos en tiempo real
- ✅ 3 Modelos de Machine Learning:
  - Random Forest (R² = 0.989)
  - Gradient Boosting (R² = 0.992)
  - Neural Network (R² = 0.944)
- ✅ Dashboard interactivo con Streamlit
- ✅ Visualizaciones en tiempo real con Plotly

## 📊 Demo

[Link al Dashboard en Vivo](https://tu-dashboard.streamlit.app) *(se agregará después del deploy)*

## 🛠️ Instalación Local
```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/tu-repo.git
cd tu-repo

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
# Crear archivo .env con tu COSMOS_CONNECTION_STRING

# Ejecutar dashboard
streamlit run dashboard.py
```

## 📁 Estructura del Proyecto
```
ProyectoAzure/
├── dashboard.py              # Dashboard interactivo
├── models.py                 # Modelos de ML
├── data_collector.py         # Recolector de datos
├── azure_connector.py        # Conector a Azure
├── auto_collector.py         # Recolección automática
├── requirements.txt          # Dependencias
├── .env                      # Variables de entorno (no incluido)
└── README.md                 # Este archivo
```

## 🎓 Proyecto Académico

Proyecto 2: Cloud Document Database with Predictive Analytics
- Azure Cosmos DB con MongoDB API
- Modelos de predicción de precios
- Dashboard interactivo en tiempo real

## 👥 Autor

Vanessa Lizcano - Universidad [Tu Universidad]
