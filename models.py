import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

from azure_connector import AzureCosmosConnector

class PricePredictor:
    """
    Sistema de predicción de precios con múltiples modelos
    """
    
    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.results = {}
        
    def cargar_datos_desde_azure(self):
        """Carga datos desde Azure Cosmos DB"""
        print("="*70)
        print("📊 CARGANDO DATOS DESDE AZURE COSMOS DB")
        print("="*70)
        
        # Conectar a Azure
        connector = AzureCosmosConnector()
        if not connector.conectar():
            raise Exception("No se pudo conectar a Azure")
        
        # Obtener todos los documentos
        print("\n📥 Descargando documentos...")
        documentos = list(connector.collection.find({}))
        
        print(f"✅ {len(documentos)} documentos descargados")
        
        # Convertir a DataFrame
        self.df = pd.DataFrame(documentos)
        
        # Cerrar conexión
        connector.cerrar_conexion()
        
        print(f"\n📋 Dimensiones del dataset: {self.df.shape}")
        print(f"📑 Columnas: {list(self.df.columns)}")
        
        return self.df
    
    def preparar_datos(self):
        """Prepara los datos para el modelado de predicción de precios"""
        print("\n" + "="*70)
        print("🔧 PREPARACIÓN DE DATOS")
        print("="*70)
        
        # Seleccionar features relevantes para PREDECIR PRECIO
        features = [
            'precio_original',
            'category',
            'stock',
            'views_hora',
            'en_carrito',
            'descuento',
            'disponible',
            'ventas_dia'
        ]
        
        # Agregar rating si existe
        if 'rating' in self.df.columns:
            # Si rating es un dict, extraer valores
            if isinstance(self.df['rating'].iloc[0], dict):
                self.df['rating_score'] = self.df['rating'].apply(lambda x: x.get('rate', 0))
                self.df['rating_count'] = self.df['rating'].apply(lambda x: x.get('count', 0))
                features.extend(['rating_score', 'rating_count'])
        
        # Target variable (lo que queremos predecir)
        target = 'precio_actual'
        
        # Crear dataset de trabajo
        print(f"\n📊 Features seleccionadas: {len(features)}")
        for f in features:
            print(f"   - {f}")
        print(f"\n🎯 Target: {target}")
        
        # Verificar que todas las columnas existen
        features_disponibles = [f for f in features if f in self.df.columns]
        print(f"\n✅ Features disponibles: {len(features_disponibles)}/{len(features)}")
        
        # Crear X y y
        X = self.df[features_disponibles].copy()
        y = self.df[target].copy()
        
        # Codificar variables categóricas
        print("\n🔤 Codificando variables categóricas...")
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
            print(f"   - {col}: {len(le.classes_)} categorías")
        
        # Convertir booleanos a int
        bool_cols = X.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            X[col] = X[col].astype(int)
        
        # Dividir datos: 70% train, 15% validation, 15% test
        print("\n✂️  Dividiendo datos...")
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=0.15, random_state=42
        )
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=42  # 0.176 de 0.85 ≈ 0.15 del total
        )
        
        print(f"   - Train: {len(self.X_train)} muestras ({len(self.X_train)/len(X)*100:.1f}%)")
        print(f"   - Validation: {len(self.X_val)} muestras ({len(self.X_val)/len(X)*100:.1f}%)")
        print(f"   - Test: {len(self.X_test)} muestras ({len(self.X_test)/len(X)*100:.1f}%)")
        
        # Mostrar estadísticas del target
        print(f"\n💰 Estadísticas del precio actual:")
        print(f"   - Min: ${y.min():.2f}")
        print(f"   - Max: ${y.max():.2f}")
        print(f"   - Media: ${y.mean():.2f}")
        print(f"   - Desv. Std: ${y.std():.2f}")
        
        # Escalar features
        print("\n📏 Escalando features...")
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("✅ Datos preparados correctamente")
        
        return self.X_train, self.y_train
    
    def entrenar_random_forest(self):
        """Entrena modelo Random Forest"""
        print("\n" + "="*70)
        print("🌲 MODELO 1: RANDOM FOREST REGRESSOR")
        print("="*70)
        
        print("\n⚙️  Configuración:")
        print("   - n_estimators: 100 árboles")
        print("   - max_depth: 15")
        print("   - min_samples_split: 5")
        print("   - random_state: 42")
        
        # Crear y entrenar modelo
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        print("\n🏋️  Entrenando...")
        rf_model.fit(self.X_train, self.y_train)
        
        # Predicciones
        y_train_pred = rf_model.predict(self.X_train)
        y_val_pred = rf_model.predict(self.X_val)
        y_test_pred = rf_model.predict(self.X_test)
        
        # Evaluar
        self.results['Random Forest'] = {
            'model': rf_model,
            'train_mae': mean_absolute_error(self.y_train, y_train_pred),
            'val_mae': mean_absolute_error(self.y_val, y_val_pred),
            'test_mae': mean_absolute_error(self.y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
            'val_rmse': np.sqrt(mean_squared_error(self.y_val, y_val_pred)),
            'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
            'train_r2': r2_score(self.y_train, y_train_pred),
            'val_r2': r2_score(self.y_val, y_val_pred),
            'test_r2': r2_score(self.y_test, y_test_pred)
        }
        
        print("\n📊 Resultados:")
        print(f"   Train - MAE: ${self.results['Random Forest']['train_mae']:.2f}, R²: {self.results['Random Forest']['train_r2']:.4f}")
        print(f"   Val   - MAE: ${self.results['Random Forest']['val_mae']:.2f}, R²: {self.results['Random Forest']['val_r2']:.4f}")
        print(f"   Test  - MAE: ${self.results['Random Forest']['test_mae']:.2f}, R²: {self.results['Random Forest']['test_r2']:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🔝 Top 5 Features más importantes:")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
        
        self.models['Random Forest'] = rf_model
        
        return rf_model
    
    def entrenar_gradient_boosting(self):
        """Entrena modelo Gradient Boosting"""
        print("\n" + "="*70)
        print("🚀 MODELO 2: GRADIENT BOOSTING REGRESSOR")
        print("="*70)
        
        print("\n⚙️  Configuración:")
        print("   - n_estimators: 100")
        print("   - learning_rate: 0.1")
        print("   - max_depth: 4")
        print("   - subsample: 0.8")
        
        # Crear y entrenar modelo
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.8,
            random_state=42
        )
        
        print("\n🏋️  Entrenando...")
        gb_model.fit(self.X_train, self.y_train)
        
        # Predicciones
        y_train_pred = gb_model.predict(self.X_train)
        y_val_pred = gb_model.predict(self.X_val)
        y_test_pred = gb_model.predict(self.X_test)
        
        # Evaluar
        self.results['Gradient Boosting'] = {
            'model': gb_model,
            'train_mae': mean_absolute_error(self.y_train, y_train_pred),
            'val_mae': mean_absolute_error(self.y_val, y_val_pred),
            'test_mae': mean_absolute_error(self.y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
            'val_rmse': np.sqrt(mean_squared_error(self.y_val, y_val_pred)),
            'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
            'train_r2': r2_score(self.y_train, y_train_pred),
            'val_r2': r2_score(self.y_val, y_val_pred),
            'test_r2': r2_score(self.y_test, y_test_pred)
        }
        
        print("\n📊 Resultados:")
        print(f"   Train - MAE: ${self.results['Gradient Boosting']['train_mae']:.2f}, R²: {self.results['Gradient Boosting']['train_r2']:.4f}")
        print(f"   Val   - MAE: ${self.results['Gradient Boosting']['val_mae']:.2f}, R²: {self.results['Gradient Boosting']['val_r2']:.4f}")
        print(f"   Test  - MAE: ${self.results['Gradient Boosting']['test_mae']:.2f}, R²: {self.results['Gradient Boosting']['test_r2']:.4f}")
        
        self.models['Gradient Boosting'] = gb_model
        
        return gb_model
    
    def entrenar_red_neuronal(self, epochs=100):
        """Entrena Red Neuronal con TensorFlow"""
        print("\n" + "="*70)
        print("🧠 MODELO 3: RED NEURONAL (DEEP LEARNING)")
        print("="*70)
        
        # Arquitectura
        print("\n🏗️  Arquitectura:")
        print("   - Input Layer: features")
        print("   - Hidden Layer 1: 128 neuronas (ReLU) + Dropout 0.3")
        print("   - Hidden Layer 2: 64 neuronas (ReLU) + Dropout 0.2")
        print("   - Hidden Layer 3: 32 neuronas (ReLU)")
        print("   - Output Layer: 1 neurona (linear)")
        
        # Crear modelo
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(self.X_train_scaled.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        
        # Compilar
        model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        print("\n⚙️  Configuración de entrenamiento:")
        print(f"   - Optimizer: Adam")
        print(f"   - Loss: MSE")
        print(f"   - Epochs: {epochs}")
        print(f"   - Batch size: 16")
        print(f"   - Early stopping: patience=15")
        
        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        # Entrenar
        print("\n🏋️  Entrenando...")
        history = model.fit(
            self.X_train_scaled, self.y_train,
            validation_data=(self.X_val_scaled, self.y_val),
            epochs=epochs,
            batch_size=16,
            callbacks=[early_stopping],
            verbose=0
        )
        
        print(f"✅ Entrenamiento completado en {len(history.history['loss'])} epochs")
        
        # Predicciones
        y_train_pred = model.predict(self.X_train_scaled, verbose=0).flatten()
        y_val_pred = model.predict(self.X_val_scaled, verbose=0).flatten()
        y_test_pred = model.predict(self.X_test_scaled, verbose=0).flatten()
        
        # Evaluar
        self.results['Neural Network'] = {
            'model': model,
            'history': history.history,
            'train_mae': mean_absolute_error(self.y_train, y_train_pred),
            'val_mae': mean_absolute_error(self.y_val, y_val_pred),
            'test_mae': mean_absolute_error(self.y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
            'val_rmse': np.sqrt(mean_squared_error(self.y_val, y_val_pred)),
            'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
            'train_r2': r2_score(self.y_train, y_train_pred),
            'val_r2': r2_score(self.y_val, y_val_pred),
            'test_r2': r2_score(self.y_test, y_test_pred)
        }
        
        print("\n📊 Resultados:")
        print(f"   Train - MAE: ${self.results['Neural Network']['train_mae']:.2f}, R²: {self.results['Neural Network']['train_r2']:.4f}")
        print(f"   Val   - MAE: ${self.results['Neural Network']['val_mae']:.2f}, R²: {self.results['Neural Network']['val_r2']:.4f}")
        print(f"   Test  - MAE: ${self.results['Neural Network']['test_mae']:.2f}, R²: {self.results['Neural Network']['test_r2']:.4f}")
        
        self.models['Neural Network'] = model
        
        return model, history
    
    def comparar_modelos(self):
        """Compara todos los modelos entrenados"""
        print("\n" + "="*70)
        print("📊 COMPARACIÓN DE MODELOS")
        print("="*70)
        
        # Crear tabla de comparación
        comparison_data = []
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Modelo': model_name,
                'Train MAE ($)': f"{metrics['train_mae']:.2f}",
                'Val MAE ($)': f"{metrics['val_mae']:.2f}",
                'Test MAE ($)': f"{metrics['test_mae']:.2f}",
                'Train R²': f"{metrics['train_r2']:.4f}",
                'Val R²': f"{metrics['val_r2']:.4f}",
                'Test R²': f"{metrics['test_r2']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n📋 Tabla de Métricas:")
        print(comparison_df.to_string(index=False))
        
        # Determinar mejor modelo
        best_model_name = min(self.results.keys(), 
                             key=lambda x: self.results[x]['test_mae'])
        
        print(f"\n🏆 MEJOR MODELO: {best_model_name}")
        print(f"   Test MAE: ${self.results[best_model_name]['test_mae']:.2f}")
        print(f"   Test R²: {self.results[best_model_name]['test_r2']:.4f}")
        
        # Interpretación del R²
        best_r2 = self.results[best_model_name]['test_r2']
        if best_r2 > 0.9:
            print(f"   Interpretación: Excelente predicción ({best_r2*100:.1f}% de la varianza explicada)")
        elif best_r2 > 0.7:
            print(f"   Interpretación: Buena predicción ({best_r2*100:.1f}% de la varianza explicada)")
        elif best_r2 > 0.5:
            print(f"   Interpretación: Predicción moderada ({best_r2*100:.1f}% de la varianza explicada)")
        else:
            print(f"   Interpretación: Predicción limitada ({best_r2*100:.1f}% de la varianza explicada)")
        
        return comparison_df, best_model_name
    
    def generar_visualizaciones(self):
        """Genera gráficos de comparación"""
        print("\n📈 Generando visualizaciones...")
        
        # Crear figura con subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Comparación de Modelos - Predicción de Precios', fontsize=16, fontweight='bold')
        
        # 1. Comparación de MAE
        model_names = list(self.results.keys())
        test_maes = [self.results[m]['test_mae'] for m in model_names]
        
        axes[0, 0].bar(model_names, test_maes, color=['#2ecc71', '#3498db', '#e74c3c'])
        axes[0, 0].set_title('Test MAE por Modelo (menor es mejor)')
        axes[0, 0].set_ylabel('MAE ($)')
        axes[0, 0].tick_params(axis='x', rotation=15)
        
        # 2. Comparación de R²
        test_r2s = [self.results[m]['test_r2'] for m in model_names]
        
        axes[0, 1].bar(model_names, test_r2s, color=['#2ecc71', '#3498db', '#e74c3c'])
        axes[0, 1].set_title('Test R² por Modelo (mayor es mejor)')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].tick_params(axis='x', rotation=15)
        axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # 3. Train vs Val Loss (Neural Network)
        if 'Neural Network' in self.results:
            history = self.results['Neural Network']['history']
            axes[1, 0].plot(history['loss'], label='Train Loss', color='#3498db')
            axes[1, 0].plot(history['val_loss'], label='Val Loss', color='#e74c3c')
            axes[1, 0].set_title('Red Neuronal - Curvas de Aprendizaje')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss (MSE)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Comparación Train vs Test
        x = np.arange(len(model_names))
        width = 0.35
        
        train_maes = [self.results[m]['train_mae'] for m in model_names]
        
        axes[1, 1].bar(x - width/2, train_maes, width, label='Train MAE', color='#2ecc71')
        axes[1, 1].bar(x + width/2, test_maes, width, label='Test MAE', color='#e74c3c')
        axes[1, 1].set_title('Train vs Test MAE (detectar overfitting)')
        axes[1, 1].set_ylabel('MAE ($)')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(model_names, rotation=15)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("✅ Gráfico guardado: model_comparison.png")
        
        return fig


# ============================================
# EJECUCIÓN DEL SCRIPT
# ============================================

if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════╗
║              🤖 SISTEMA DE PREDICCIÓN DE PRECIOS                  ║
║                    Machine Learning Pipeline                       ║
╚════════════════════════════════════════════════════════════════════╝
    """)
    
    # Crear instancia del predictor
    predictor = PricePredictor()
    
    try:
        # 1. Cargar datos desde Azure
        predictor.cargar_datos_desde_azure()
        
        # 2. Preparar datos
        predictor.preparar_datos()
        
        # 3. Entrenar modelos
        predictor.entrenar_random_forest()
        predictor.entrenar_gradient_boosting()
        predictor.entrenar_red_neuronal(epochs=100)
        
        # 4. Comparar modelos
        comparison_df, best_model = predictor.comparar_modelos()
        
        # 5. Generar visualizaciones
        predictor.generar_visualizaciones()
        
        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
        print("="*70)
        print("\n📁 Archivos generados:")
        print("   - model_comparison.png (gráficos de comparación)")
        print("\n💡 Próximo paso: Crear dashboard interactivo con Streamlit")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()