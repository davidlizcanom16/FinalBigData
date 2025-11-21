# 🛍️ E-commerce Price Prediction Dashboard

Dashboard de análisis predictivo de precios en tiempo real usando Azure Cosmos DB y Machine Learning.

## 🌐 Demo en Vivo

**Dashboard:** [https://tu-app.streamlit.app](https://tu-app.streamlit.app) *(se actualizará después del deploy)*

## ✨ Características

- ☁️ **Base de datos en la nube**: Azure Cosmos DB con MongoDB API
- 🤖 **3 Modelos de ML**: Random Forest, Gradient Boosting, Neural Network
- 📊 **Dashboard interactivo**: Streamlit con actualización automática cada 60s
- 🔄 **Recolección automática**: GitHub Actions ejecuta cada 5 minutos
- 📈 **Visualizaciones**: Gráficos interactivos con Plotly

## 🎯 Resultados de los Modelos

| Modelo | Test MAE | Test R² |
|--------|----------|---------|
| Random Forest | $12.82 | 0.9892 |
| Gradient Boosting | $11.28 | 0.9917 |
| Neural Network | $37.86 | 0.9441 |

## 🚀 Despliegue (Recomendado)

El proyecto está configurado para desplegarse automáticamente:

1. **Fork este repositorio**
2. **Configura secretos en GitHub**:
   - Ve a Settings → Secrets → Actions
   - Agrega: `COSMOS_CONNECTION_STRING` con tu connection string de Azure
3. **Despliega en Streamlit Cloud**:
   - Ve a [share.streamlit.io](https://share.streamlit.io)
   - Conecta tu GitHub
   - Selecciona el repo y `dashboard.py`
   - Agrega el secret: `COSMOS_CONNECTION_STRING`

El dashboard estará disponible 24/7 y los datos se recolectarán automáticamente cada 5 minutos.

## 💻 Instalación Local

Si prefieres ejecutarlo localmente:

### Prerequisitos
- Python 3.12+
- Cuenta de Azure con Cosmos DB

### Pasos
```bash
# 1. Clonar repositorio
git clone https://github.com/tu-usuario/ecommerce-price-prediction.git
cd ecommerce-price-prediction

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar variables de entorno
cp .env.example .env
# Edita .env y agrega tu COSMOS_CONNECTION_STRING

# 4. Ejecutar dashboard
streamlit run dashboard.py
```

### Recolección manual de datos
```bash
# Recolectar datos una vez
python data_collector.py

# Recolección automática continua
python auto_collector.py
```

## 📁 Estructura del Proyecto
```
ecommerce-price-prediction/
├── .github/
│   └── workflows/
│       └── auto_collector.yml    # GitHub Actions para recolección
├── .streamlit/
│   └── config.toml              # Configuración de Streamlit
├── dashboard.py                  # 📊 Dashboard principal
├── models.py                     # 🤖 Modelos de ML
├── data_collector.py             # 📡 Recolector de datos
├── azure_connector.py            # ☁️ Conector a Azure
├── auto_collector.py             # 🔄 Recolección automática
├── requirements.txt              # 📦 Dependencias
├── .env.example                  # 🔐 Template de variables
├── .gitignore                   # 🚫 Archivos ignorados
└── README.md                     # 📖 Este archivo
```

## 🏗️ Arquitectura
```
┌─────────────────┐
│  Fake Store API │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────────┐
│ GitHub Actions  │─────▶│ Azure Cosmos DB  │
│ (cada 5 min)    │      │   (MongoDB API)  │
└─────────────────┘      └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Streamlit Cloud  │
                         │   (Dashboard)    │
                         └──────────────────┘
```

## 🔐 Configuración de Azure

1. Crear cuenta en [Azure Portal](https://portal.azure.com)
2. Crear Azure Cosmos DB con MongoDB API
3. Seleccionar modo "Serverless" (gratis)
4. Copiar la Connection String
5. Agregar a `.env` o como secret en GitHub/Streamlit

## 📊 Uso del Dashboard

El dashboard tiene 4 secciones:

1. **📊 Dashboard Principal**: KPIs y visualizaciones generales
2. **🤖 Predicción de Precios**: Predicción interactiva con los 3 modelos
3. **📈 Análisis de Modelos**: Comparación y métricas de rendimiento
4. **📋 Datos en Tiempo Real**: Exploración de datos con filtros

## 🎓 Contexto Académico

**Proyecto 2**: Cloud Document Database with Predictive Analytics

**Requisitos cumplidos:**
- ✅ Base de datos NoSQL en Azure
- ✅ 3+ modelos predictivos (incluyendo red neuronal)
- ✅ Dashboard interactivo
- ✅ Conexión en tiempo real a base de datos en la nube
- ✅ Documentación completa
- ✅ Presentación profesional

## 🤝 Contribuir

Si quieres contribuir:

1. Fork el proyecto
2. Crea una rama (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -m 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## 📝 Licencia

MIT License - libre para uso académico y comercial

## 👤 Autor

**Vanessa Lizcano**

Proyecto desarrollado para el curso de Cloud Computing
Universidad [Tu Universidad] - 2024

---

⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub
```

---

### 19.5 Actualizar `.gitignore`
```
# Archivos de entorno
.env

# Archivos de datos temporales
datos_ecommerce.json
*.log
collector.log

# Python
__pycache__/
*.py[cod]
*$py.class
*.so

# Jupyter Notebook
.ipynb_checkpoints

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Outputs
model_comparison.png
*.pkl
```

---

## 📦 Estructura Final de Archivos

Tu proyecto debe tener:
```
ProyectoAzure/
├── .github/
│   └── workflows/
│       └── auto_collector.yml
├── .streamlit/
│   └── config.toml
├── dashboard.py
├── models.py
├── data_collector.py
├── azure_connector.py
├── auto_collector.py
├── test_api.py
├── requirements.txt
├── .env (NO subir a GitHub)
├── .env.example (SÍ subir)
├── .gitignore
└── README.md
