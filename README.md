# E-commerce Price Prediction System

Cloud-native predictive analytics platform for real-time e-commerce pricing using Azure Cosmos DB and machine learning.

## Project Overview

This project implements a comprehensive data pipeline and machine learning system for predicting e-commerce product prices. The system leverages Azure's cloud infrastructure for scalable data storage and automated data collection, combined with multiple predictive models to achieve high accuracy in price forecasting.

**Academic Context:** Project 2 - Cloud Document Database with Predictive Analytics  
**Technologies:** Azure Cosmos DB (MongoDB API), Python, Streamlit, GitHub Actions

---

## Live Demo

**Dashboard:** [https://finalbigdata-cneccdp3sxcclrkfbqqogy.streamlit.app/](https://finalbigdata-cneccdp3sxcclrkfbqqogy.streamlit.app/)

**Repository:** [https://github.com/davidlizcanom16/FinalBigData](https://github.com/davidlizcanom16/FinalBigData)

---

## Architecture
```
┌──────────────────┐
│  Fake Store API  │
│   (Data Source)  │
└────────┬─────────┘
         │
         ▼
┌──────────────────────┐      ┌─────────────────────────┐
│   GitHub Actions     │─────▶│   Azure Cosmos DB       │
│   (Every 5 minutes)  │      │   (MongoDB API)         │
│   - Data Collection  │      │   - NoSQL Storage       │
│   - Preprocessing    │      │   - Scalable & Managed  │
└──────────────────────┘      └────────┬────────────────┘
                                       │
                                       ▼
                              ┌────────────────────────┐
                              │  Streamlit Dashboard   │
                              │  - Real-time Analytics │
                              │  - ML Predictions      │
                              │  - Interactive UI      │
                              └────────────────────────┘
```

---

## Key Features

### Cloud Infrastructure
- **Azure Cosmos DB** with MongoDB API for document storage
- **Serverless deployment** with automatic scaling
- **GitHub Actions** for continuous data collection (every 5 minutes)
- **Real-time synchronization** between cloud database and dashboard

### Machine Learning Models

Three predictive models trained and compared:

1. **Random Forest Regressor**
   - Ensemble learning with 100 decision trees
   - Test MAE: $12.82
   - Test R²: 0.9892
   - Best for: Interpretability and feature importance analysis

2. **Gradient Boosting Regressor**
   - Sequential ensemble with 100 estimators
   - Test MAE: $11.28
   - Test R²: 0.9917
   - Best performance: Highest accuracy across all metrics

3. **Feedforward Neural Network**
   - Architecture: Input → Dense(64, ReLU) → Dropout(0.3) → Dense(32, ReLU) → Dropout(0.2) → Output(1)
   - Optimizer: Adam
   - Loss function: Mean Squared Error
   - Training: 50 epochs with validation split
   - Test MAE: $37.86
   - Test R²: 0.9441
   - Best for: Handling non-linear relationships

### Interactive Dashboard

Four main sections:
- **Main Dashboard**: KPIs, price distributions, temporal analysis
- **Price Prediction**: Interactive prediction with all three models
- **Model Analysis**: Performance metrics and feature importance
- **Real-time Data**: Filterable product catalog with live updates

---

## Model Performance

| Model | Test MAE ($) | Test RMSE ($) | Test R² | Validation MAE ($) | Validation R² |
|-------|--------------|---------------|---------|-------------------|---------------|
| Random Forest | 12.82 | 23.67 | 0.9892 | 13.45 | 0.9876 |
| **Gradient Boosting** | **11.28** | **21.34** | **0.9917** | **12.01** | **0.9905** |
| Neural Network | 37.86 | 58.23 | 0.9441 | 39.12 | 0.9387 |

**Winner:** Gradient Boosting achieves the best performance with 99.17% variance explained and lowest error metrics.

---

## Technical Implementation

### Data Pipeline

**1. Data Collection (github_collector_cached.py)**
```python
- Generates 20 products per execution
- Simulates price variations (±10%)
- Includes real-time metrics: stock, sales, views, cart additions
- Runs automatically via GitHub Actions every 5 minutes
```

**2. Data Storage (Azure Cosmos DB)**
```
- Database: ecommerce_db
- Collection: productos
- Document structure: JSON with embedded metadata
- Current size: ~1,500+ documents
```

**3. Feature Engineering**
```python
Features used for prediction:
- precio_original: Base product price
- category: Product category (encoded)
- stock: Available inventory
- views_hora: Hourly page views
- en_carrito: Items in shopping carts
- descuento: Discount percentage
- disponible: Availability status
- ventas_dia: Daily sales count
- rating_score: Customer rating (1-5)
- rating_count: Number of reviews
```

### Model Training Pipeline
```python
1. Data Loading from Azure Cosmos DB
2. Data Cleaning and Preprocessing
3. Feature Encoding (Label Encoding for categorical variables)
4. Train/Test Split (80/20)
5. Feature Scaling (StandardScaler for Neural Network)
6. Model Training with cross-validation
7. Hyperparameter tuning
8. Performance evaluation
```

---

## Installation and Usage

### Prerequisites

- Python 3.11+
- Azure account with Cosmos DB access
- GitHub account (for automated collection)

### Local Setup
```bash
# Clone repository
git clone https://github.com/davidlizcanom16/FinalBigData.git
cd FinalBigData

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and add your COSMOS_CONNECTION_STRING

# Run dashboard
streamlit run dashboard.py
```

### Azure Cosmos DB Configuration

1. Create Azure Cosmos DB account
2. Select **MongoDB API**
3. Choose **Serverless** capacity mode (cost-effective)
4. Configure networking:
   - Enable public access
   - Add IP ranges: `0.0.0.0/0` (for development)
5. Copy connection string to `.env`

### GitHub Actions Setup

1. Go to repository **Settings** → **Secrets and variables** → **Actions**
2. Add secret: `COSMOS_CONNECTION_STRING`
3. Workflow runs automatically every 5 minutes
4. Manual trigger available via Actions tab

### Streamlit Cloud Deployment

1. Visit [share.streamlit.io](https://share.streamlit.io)
2. Connect GitHub repository
3. Set main file: `dashboard.py`
4. Add secret in **Advanced settings**:
```toml
   COSMOS_CONNECTION_STRING = "your-connection-string-here"
```
5. Deploy

---

## Project Structure
```
FinalBigData/
├── .github/
│   └── workflows/
│       └── auto_collector.yml          # GitHub Actions workflow
├── .streamlit/
│   └── config.toml                     # Streamlit configuration
├── dashboard.py                         # Main dashboard application
├── models.py                            # ML model implementations
├── data_collector.py                    # Data collection from API
├── azure_connector.py                   # Azure Cosmos DB interface
├── github_collector_cached.py           # Automated collection script
├── auto_collector.py                    # Local collection utility
├── requirements.txt                     # Python dependencies
├── .env.example                         # Environment template
├── .gitignore                          # Git ignore rules
└── README.md                           # This file
```

---

## Data Schema

### Product Document Structure
```json
{
  "_id": "ObjectId",
  "id": 1,
  "title": "Product Name",
  "price": 109.95,
  "category": "electronics",
  "rating": {
    "rate": 3.9,
    "count": 120
  },
  "timestamp": "2024-11-21T18:00:00",
  "precio_original": 109.95,
  "precio_actual": 115.23,
  "cambio_precio_porcentaje": 4.8,
  "stock": 250,
  "ventas_dia": 45,
  "views_hora": 1250,
  "en_carrito": 12,
  "disponible": true,
  "descuento": 0,
  "rating_score": 3.9,
  "rating_count": 120,
  "snapshot_id": "auto",
  "snapshot_timestamp": "2024-11-21T18:00:00",
  "github_action": true
}
```

---

## API Reference

### Azure Cosmos DB Connection
```python
from azure_connector import AzureCosmosConnector

connector = AzureCosmosConnector()
if connector.conectar():
    # Query documents
    documents = connector.collection.find({})
    
    # Insert document
    connector.collection.insert_one(document)
    
    # Close connection
    connector.cerrar_conexion()
```

### Price Prediction
```python
from models import PricePredictor

predictor = PricePredictor()
predictor.preparar_datos()
predictor.entrenar_random_forest()

# Make prediction
features = [precio_original, categoria_encoded, stock, ...]
prediction = predictor.models['Random Forest'].predict([features])
```

---

## Performance Optimization

### Database
- Indexed fields: `id`, `category`, `snapshot_id`
- Connection pooling enabled
- Serverless scaling for cost optimization

### Models
- Feature scaling for neural networks
- Early stopping to prevent overfitting
- Dropout layers (0.2-0.3) for regularization
- Hyperparameter tuning via grid search

### Dashboard
- Caching with 5-minute TTL
- Lazy loading for model training
- Efficient data aggregation queries
- Auto-refresh mechanism (60-second interval)

---

## Future Enhancements

1. **Advanced Models**
   - LSTM networks for time series forecasting
   - XGBoost ensemble methods
   - Hyperparameter optimization with Optuna

2. **Real-time Features**
   - WebSocket connections for live updates
   - Streaming data pipeline
   - Real-time anomaly detection

3. **Production Features**
   - A/B testing framework
   - Model versioning and monitoring
   - Automated retraining pipeline
   - Alert system for price anomalies

4. **Extended Analytics**
   - Customer segmentation
   - Demand forecasting
   - Competitive pricing analysis
   - Seasonal trend detection

---

## Security Considerations

- Connection strings stored in environment variables
- GitHub Secrets for CI/CD pipeline
- Azure RBAC for database access
- Network security via IP whitelisting
- No sensitive data in version control

---

## Contributing

Contributions are welcome. Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/enhancement`)
3. Commit changes (`git commit -m 'Add enhancement'`)
4. Push to branch (`git push origin feature/enhancement`)
5. Open a Pull Request

---

## License

MIT License - Free for academic and commercial use

---

## Author

**David Lizcano**  
Master's Student - Data Science and Cloud Computing  
Universidad [Your University]

---

## Acknowledgments

- Azure Cosmos DB documentation and support
- Fake Store API for test data
- Streamlit community for dashboard framework
- GitHub Actions for CI/CD automation

---

## Contact

For questions or collaboration:
- GitHub: [@davidlizcanom16](https://github.com/davidlizcanom16)
- Repository: [FinalBigData](https://github.com/davidlizcanom16/FinalBigData)

---

**Project Status:** Production Ready  
**Last Updated:** November 21, 2024  
**Version:** 1.0.0
