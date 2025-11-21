import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import time

from azure_connector import AzureCosmosConnector
from models import PricePredictor

# Configuración de la página
st.set_page_config(
    page_title="E-commerce Price Analytics Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 20px;
    }
    h2 {
        color: #2c3e50;
        padding-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Funciones de caché
@st.cache_data(ttl=300)
def cargar_datos_azure():
    """Carga datos desde Azure Cosmos DB"""
    connector = AzureCosmosConnector()
    if connector.conectar():
        documentos = list(connector.collection.find({}))
        connector.cerrar_conexion()
        
        if len(documentos) == 0:
            return None
        
        df = pd.DataFrame(documentos)
        
        # Procesar rating si es dict
        if 'rating' in df.columns and len(df) > 0:
            if isinstance(df['rating'].iloc[0], dict):
                df['rating_score'] = df['rating'].apply(lambda x: x.get('rate', 0))
                df['rating_count'] = df['rating'].apply(lambda x: x.get('count', 0))
        
        return df
    return None

@st.cache_resource
def cargar_modelos():
    """Carga o entrena los modelos de ML"""
    predictor = PricePredictor()
    df = cargar_datos_azure()
    if df is None:
        return None
    
    predictor.df = df
    predictor.preparar_datos()
    predictor.entrenar_random_forest()
    predictor.entrenar_gradient_boosting()
    predictor.entrenar_red_neuronal(epochs=50)
    
    return predictor

def predecir_precio(predictor, features):
    """Hace predicción con los 3 modelos"""
    feature_array = np.array([features])
    feature_scaled = predictor.scaler.transform(feature_array)
    
    predicciones = {
        'Random Forest': predictor.models['Random Forest'].predict(feature_array)[0],
        'Gradient Boosting': predictor.models['Gradient Boosting'].predict(feature_array)[0],
        'Neural Network': predictor.models['Neural Network'].predict(feature_scaled, verbose=0)[0][0]
    }
    
    return predicciones

# Header
st.title("🛍️ E-commerce Price Analytics Dashboard")
st.markdown("**Análisis en Tiempo Real con Azure Cosmos DB + Machine Learning**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Panel de Control")
    
    # Botón de actualización manual
    if st.button("🔄 Actualizar Datos", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    
    # Selector de página
    pagina = st.radio(
        "Navegación:",
        ["📊 Dashboard Principal", "🤖 Predicción de Precios", "📈 Análisis de Modelos", "📋 Datos en Tiempo Real"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### 📌 Información")
    st.info("Dashboard conectado a Azure Cosmos DB en tiempo real.")

# Cargar datos
with st.spinner("🔄 Cargando datos desde Azure..."):
    df = cargar_datos_azure()

if df is None:
    st.error("❌ No se pudo conectar a Azure Cosmos DB")
    st.stop()

# ============================================
# PÁGINA 1: DASHBOARD PRINCIPAL
# ============================================

if pagina == "📊 Dashboard Principal":
    
    # KPIs principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📦 Total Productos", f"{len(df['id'].unique())}", delta="En inventario")
    
    with col2:
        st.metric("💰 Precio Promedio", f"${df['precio_actual'].mean():.2f}", delta=f"±${df['precio_actual'].std():.2f}")
    
    with col3:
        st.metric("📸 Snapshots", f"{df['snapshot_id'].nunique()}", delta="Actualizaciones")
    
    with col4:
        disponibles = df['disponible'].sum()
        st.metric("✅ Disponibles", f"{disponibles}/{len(df)}", delta=f"{(disponibles/len(df)*100):.1f}%")
    
    st.markdown("---")
    
    # Gráficos principales
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Distribución de Precios por Categoría")
        fig = px.box(df, x='category', y='precio_actual', color='category', title="Análisis de Precios por Categoría")
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Variación de Precios en el Tiempo")
        precio_tiempo = df.groupby('snapshot_id')['precio_actual'].mean().reset_index()
        fig = px.line(precio_tiempo, x='snapshot_id', y='precio_actual', markers=True, title="Evolución del Precio Promedio")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏷️ Productos por Categoría")
        categoria_counts = df.groupby('category')['id'].nunique().reset_index()
        categoria_counts.columns = ['category', 'count']
        fig = px.pie(categoria_counts, names='category', values='count', title="Distribución de Productos", hole=0.4)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⭐ Rating vs Precio")
        fig = px.scatter(df, x='rating_score', y='precio_actual', color='category', size='rating_count', 
                        hover_data=['title'], title="Relación entre Rating y Precio")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🔝 Top 10 Productos Más Caros")
    
    top_productos = df.nlargest(10, 'precio_actual')[['title', 'category', 'precio_actual', 'rating_score', 'stock', 'disponible']].copy()
    top_productos['precio_actual'] = top_productos['precio_actual'].apply(lambda x: f"${x:.2f}")
    top_productos.columns = ['Producto', 'Categoría', 'Precio', 'Rating', 'Stock', 'Disponible']
    st.dataframe(top_productos, use_container_width=True, hide_index=True)

# ============================================
# PÁGINA 2: PREDICCIÓN DE PRECIOS
# ============================================

elif pagina == "🤖 Predicción de Precios":
    
    st.header("🤖 Predicción de Precios con Machine Learning")
    st.markdown("Usa los 3 modelos entrenados para predecir el precio de un producto")
    
    with st.spinner("🔄 Cargando modelos de ML..."):
        predictor = cargar_modelos()
    
    if predictor is None:
        st.error("❌ No se pudieron cargar los modelos")
        st.stop()
    
    st.success("✅ Modelos cargados: Random Forest, Gradient Boosting, Neural Network")
    st.markdown("---")
    
    st.subheader("📝 Características del Producto")
    
    col1, col2 = st.columns(2)
    
    with col1:
        precio_original = st.number_input("Precio Original ($)", min_value=1.0, max_value=2000.0, value=100.0, step=10.0)
        categoria = st.selectbox("Categoría", options=['electronics', 'jewelery', "men's clothing", "women's clothing"])
        stock = st.slider("Stock Disponible", min_value=0, max_value=500, value=250)
        descuento = st.slider("Descuento (%)", min_value=0, max_value=50, value=0, step=5)
    
    with col2:
        rating_score = st.slider("Rating (⭐)", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
        rating_count = st.number_input("Número de Reviews", min_value=0, max_value=1000, value=100, step=10)
        views_hora = st.number_input("Views por Hora", min_value=0, max_value=10000, value=1000, step=100)
        ventas_dia = st.number_input("Ventas por Día", min_value=0, max_value=200, value=50, step=5)
    
    col1, col2 = st.columns(2)
    with col1:
        en_carrito = st.number_input("Productos en Carrito", min_value=0, max_value=100, value=10, step=1)
    with col2:
        disponible = st.selectbox("Disponible", options=[True, False], format_func=lambda x: "Sí" if x else "No")
    
    if st.button("🔮 Predecir Precio", type="primary", use_container_width=True):
        
        categoria_encoded = predictor.label_encoders['category'].transform([categoria])[0]
        features = [precio_original, categoria_encoded, stock, views_hora, en_carrito, descuento, 
                   int(disponible), ventas_dia, rating_score, rating_count]
        
        predicciones = predecir_precio(predictor, features)
        
        st.markdown("---")
        st.subheader("📊 Resultados de la Predicción")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🌲 Random Forest")
            st.markdown(f"<h2 style='color: #2ecc71;'>${predicciones['Random Forest']:.2f}</h2>", unsafe_allow_html=True)
            st.caption(f"MAE: ${predictor.results['Random Forest']['test_mae']:.2f}")
        
        with col2:
            st.markdown("### 🚀 Gradient Boosting")
            st.markdown(f"<h2 style='color: #3498db;'>${predicciones['Gradient Boosting']:.2f}</h2>", unsafe_allow_html=True)
            st.caption(f"MAE: ${predictor.results['Gradient Boosting']['test_mae']:.2f}")
        
        with col3:
            st.markdown("### 🧠 Neural Network")
            st.markdown(f"<h2 style='color: #e74c3c;'>${predicciones['Neural Network']:.2f}</h2>", unsafe_allow_html=True)
            st.caption(f"MAE: ${predictor.results['Neural Network']['test_mae']:.2f}")
        
        precio_promedio = np.mean(list(predicciones.values()))
        st.markdown("---")
        st.markdown(f"### 💰 Precio Recomendado (Promedio): **${precio_promedio:.2f}**")
        
        diferencia = precio_promedio - precio_original
        porcentaje = (diferencia / precio_original) * 100
        
        if diferencia > 0:
            st.success(f"📈 El modelo sugiere un precio {porcentaje:.1f}% mayor (${diferencia:.2f})")
        elif diferencia < 0:
            st.warning(f"📉 El modelo sugiere un precio {abs(porcentaje):.1f}% menor (${abs(diferencia):.2f})")
        else:
            st.info("➡️ El precio sugerido es similar al original")

# ============================================
# PÁGINA 3: ANÁLISIS DE MODELOS
# ============================================

elif pagina == "📈 Análisis de Modelos":
    
    st.header("📈 Análisis y Comparación de Modelos")
    
    with st.spinner("🔄 Cargando modelos..."):
        predictor = cargar_modelos()
    
    if predictor is None:
        st.error("❌ No se pudieron cargar los modelos")
        st.stop()
    
    st.subheader("📊 Métricas de Rendimiento")
    
    metrics_data = []
    for model_name, metrics in predictor.results.items():
        metrics_data.append({
            'Modelo': model_name,
            'Test MAE ($)': f"${metrics['test_mae']:.2f}",
            'Test RMSE ($)': f"${metrics['test_rmse']:.2f}",
            'Test R²': f"{metrics['test_r2']:.4f}",
            'Val MAE ($)': f"${metrics['val_mae']:.2f}",
            'Val R²': f"{metrics['val_r2']:.4f}"
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    best_model = min(predictor.results.keys(), key=lambda x: predictor.results[x]['test_mae'])
    st.success(f"🏆 **Mejor Modelo**: {best_model} con Test MAE de ${predictor.results[best_model]['test_mae']:.2f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Comparación de MAE")
        model_names = list(predictor.results.keys())
        test_maes = [predictor.results[m]['test_mae'] for m in model_names]
        
        fig = go.Figure(data=[go.Bar(x=model_names, y=test_maes, marker_color=['#2ecc71', '#3498db', '#e74c3c'],
                                     text=[f"${mae:.2f}" for mae in test_maes], textposition='outside')])
        fig.update_layout(title="Test MAE por Modelo (menor es mejor)", yaxis_title="MAE ($)", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Comparación de R²")
        test_r2s = [predictor.results[m]['test_r2'] for m in model_names]
        
        fig = go.Figure(data=[go.Bar(x=model_names, y=test_r2s, marker_color=['#2ecc71', '#3498db', '#e74c3c'],
                                     text=[f"{r2:.4f}" for r2 in test_r2s], textposition='outside')])
        fig.update_layout(title="Test R² por Modelo (mayor es mejor)", yaxis_title="R² Score", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🔝 Feature Importance (Random Forest)")
    
    rf_model = predictor.models['Random Forest']
    feature_importance = pd.DataFrame({
        'Feature': predictor.X_train.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h', 
                title="Importancia de Características en la Predicción")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    if 'Neural Network' in predictor.results:
        st.markdown("---")
        st.subheader("📈 Curvas de Aprendizaje - Red Neuronal")
        
        history = predictor.results['Neural Network']['history']
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Loss', 'MAE'))
        
        fig.add_trace(go.Scatter(y=history['loss'], name='Train Loss', line=dict(color='#3498db')), row=1, col=1)
        fig.add_trace(go.Scatter(y=history['val_loss'], name='Val Loss', line=dict(color='#e74c3c')), row=1, col=1)
        fig.add_trace(go.Scatter(y=history['mae'], name='Train MAE', line=dict(color='#3498db')), row=1, col=2)
        fig.add_trace(go.Scatter(y=history['val_mae'], name='Val MAE', line=dict(color='#e74c3c')), row=1, col=2)
        
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="MAE", row=1, col=2)
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# PÁGINA 4: DATOS EN TIEMPO REAL
# ============================================

elif pagina == "📋 Datos en Tiempo Real":
    
    st.header("📋 Datos en Tiempo Real desde Azure Cosmos DB")
    st.subheader("🔍 Filtros")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        categorias_seleccionadas = st.multiselect("Categorías", options=df['category'].unique(), default=df['category'].unique())
    
    with col2:
        rango_precio = st.slider("Rango de Precio ($)", min_value=float(df['precio_actual'].min()), 
                                max_value=float(df['precio_actual'].max()), 
                                value=(float(df['precio_actual'].min()), float(df['precio_actual'].max())))
    
    with col3:
        disponibilidad = st.selectbox("Disponibilidad", options=["Todos", "Solo Disponibles", "Solo No Disponibles"])
    
    df_filtrado = df[df['category'].isin(categorias_seleccionadas)]
    df_filtrado = df_filtrado[(df_filtrado['precio_actual'] >= rango_precio[0]) & (df_filtrado['precio_actual'] <= rango_precio[1])]
    
    if disponibilidad == "Solo Disponibles":
        df_filtrado = df_filtrado[df_filtrado['disponible'] == True]
    elif disponibilidad == "Solo No Disponibles":
        df_filtrado = df_filtrado[df_filtrado['disponible'] == False]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📦 Productos Filtrados", len(df_filtrado))
    with col2:
        st.metric("💰 Precio Promedio", f"${df_filtrado['precio_actual'].mean():.2f}")
    with col3:
        st.metric("📊 Ventas Promedio/Día", f"{df_filtrado['ventas_dia'].mean():.1f}")
    
    st.markdown("---")
    st.subheader("📊 Tabla de Productos")
    
    columnas_mostrar = st.multiselect("Selecciona columnas", 
                                     options=['title', 'category', 'precio_original', 'precio_actual', 'rating_score', 
                                             'stock', 'ventas_dia', 'views_hora', 'descuento', 'disponible'],
                                     default=['title', 'category', 'precio_actual', 'rating_score', 'stock', 'disponible'])
    
    if columnas_mostrar:
        tabla_mostrar = df_filtrado[columnas_mostrar].copy()
        
        if 'precio_actual' in columnas_mostrar:
            tabla_mostrar['precio_actual'] = tabla_mostrar['precio_actual'].apply(lambda x: f"${x:.2f}")
        if 'precio_original' in columnas_mostrar:
            tabla_mostrar['precio_original'] = tabla_mostrar['precio_original'].apply(lambda x: f"${x:.2f}")
        
        st.dataframe(tabla_mostrar, use_container_width=True, height=400)
        
        csv = tabla_mostrar.to_csv(index=False)
        st.download_button(label="📥 Descargar CSV", data=csv, file_name="productos_filtrados.csv", mime="text/csv")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 20px;'>
    <p>🚀 <strong>E-commerce Analytics Dashboard</strong> | Powered by Azure Cosmos DB + Machine Learning</p>
    <p>📊 Proyecto 2: Cloud Document Database with Predictive Analytics</p>
</div>
""", unsafe_allow_html=True)
