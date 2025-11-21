#!/usr/bin/env python3
"""
Script para GitHub Actions - Genera datos con variaciones sin llamar a API externa
"""

import pymongo
import random
from datetime import datetime
from dotenv import load_dotenv
import os
import sys

load_dotenv()

def generar_productos_con_variacion():
    """
    Genera 20 productos con variaciones de precio basados en datos base
    Sin necesidad de llamar a la API externa
    """
    
    # Productos base (simulados, basados en Fake Store)
    productos_base = [
        {"id": 1, "title": "Fjallraven Backpack", "price": 109.95, "category": "men's clothing", "rating": {"rate": 3.9, "count": 120}},
        {"id": 2, "title": "Mens Casual T-Shirts", "price": 22.3, "category": "men's clothing", "rating": {"rate": 4.1, "count": 259}},
        {"id": 3, "title": "Mens Cotton Jacket", "price": 55.99, "category": "men's clothing", "rating": {"rate": 4.7, "count": 500}},
        {"id": 4, "title": "Mens Casual Slim Fit", "price": 15.99, "category": "men's clothing", "rating": {"rate": 2.1, "count": 430}},
        {"id": 5, "title": "John Hardy Bracelet", "price": 695, "category": "jewelery", "rating": {"rate": 4.6, "count": 400}},
        {"id": 6, "title": "Solid Gold Petite", "price": 168, "category": "jewelery", "rating": {"rate": 3.9, "count": 70}},
        {"id": 7, "title": "White Gold Plated", "price": 9.99, "category": "jewelery", "rating": {"rate": 3, "count": 400}},
        {"id": 8, "title": "Pierced Owl Rose Gold", "price": 10.99, "category": "jewelery", "rating": {"rate": 1.9, "count": 100}},
        {"id": 9, "title": "WD 2TB External Hard Drive", "price": 64, "category": "electronics", "rating": {"rate": 3.3, "count": 203}},
        {"id": 10, "title": "SanDisk SSD 1TB", "price": 109, "category": "electronics", "rating": {"rate": 2.9, "count": 470}},
        {"id": 11, "title": "Silicon Power 256GB", "price": 109, "category": "electronics", "rating": {"rate": 4.8, "count": 319}},
        {"id": 12, "title": "WD 4TB Gaming Drive", "price": 114, "category": "electronics", "rating": {"rate": 4.8, "count": 400}},
        {"id": 13, "title": "Acer SB220Q Monitor", "price": 599, "category": "electronics", "rating": {"rate": 2.9, "count": 250}},
        {"id": 14, "title": "Samsung 49-Inch Monitor", "price": 999.99, "category": "electronics", "rating": {"rate": 2.2, "count": 140}},
        {"id": 15, "title": "BIYLACLESEN Jacket", "price": 56.99, "category": "women's clothing", "rating": {"rate": 2.6, "count": 235}},
        {"id": 16, "title": "Lock and Love Jacket", "price": 29.95, "category": "women's clothing", "rating": {"rate": 2.9, "count": 340}},
        {"id": 17, "title": "Rain Jacket Windbreaker", "price": 39.99, "category": "women's clothing", "rating": {"rate": 3.8, "count": 679}},
        {"id": 18, "title": "MBJ Short Sleeve Boat", "price": 9.85, "category": "women's clothing", "rating": {"rate": 4.7, "count": 130}},
        {"id": 19, "title": "Opna Short Sleeve", "price": 7.95, "category": "women's clothing", "rating": {"rate": 4.5, "count": 146}},
        {"id": 20, "title": "DANVOUY Womens T Shirt", "price": 12.99, "category": "women's clothing", "rating": {"rate": 3.6, "count": 145}}
    ]
    
    timestamp = datetime.now().isoformat()
    productos_enriquecidos = []
    
    for producto in productos_base:
        # Variación de precio (±10%)
        variacion = random.uniform(-0.10, 0.10)
        precio_actual = producto['price'] * (1 + variacion)
        
        producto_enriquecido = {
            'id': producto['id'],
            'title': producto['title'],
            'price': producto['price'],
            'category': producto['category'],
            'rating': producto['rating'],
            'timestamp': timestamp,
            'precio_original': producto['price'],
            'precio_actual': round(precio_actual, 2),
            'cambio_precio_porcentaje': round(variacion * 100, 2),
            'stock': random.randint(10, 500),
            'ventas_dia': random.randint(0, 100),
            'views_hora': random.randint(50, 5000),
            'en_carrito': random.randint(0, 50),
            'disponible': random.choice([True, True, True, False]),
            'descuento': random.choice([0, 0, 0, 5, 10, 15, 20]),
            'rating_score': producto['rating']['rate'],
            'rating_count': producto['rating']['count']
        }
        
        productos_enriquecidos.append(producto_enriquecido)
    
    return productos_enriquecidos, timestamp

def main():
    try:
        print("🚀 Iniciando recolección de datos (modo cached - sin API externa)...")
        
        connection_string = os.getenv('COSMOS_CONNECTION_STRING')
        if not connection_string:
            print("❌ Error: COSMOS_CONNECTION_STRING no encontrado")
            sys.exit(1)
        
        # Conectar a Azure
        print("🔌 Conectando a Azure Cosmos DB...")
        client = pymongo.MongoClient(connection_string)
        db = client['ecommerce_db']
        collection = db['productos']
        
        # Verificar conexión
        client.server_info()
        print("✅ Conectado a Azure Cosmos DB")
        
        # Generar productos con variaciones
        productos, timestamp = generar_productos_con_variacion()
        print(f"📦 Generados {len(productos)} productos con variaciones de precio")
        
        # Subir a Azure
        insertados = 0
        for producto in productos:
            producto['snapshot_timestamp'] = timestamp
            producto['github_action'] = True
            collection.insert_one(producto)
            insertados += 1
        
        print(f"✅ {insertados} productos insertados en Azure Cosmos DB")
        
        # Estadísticas
        total_docs = collection.count_documents({})
        print(f"📊 Total de documentos en Azure: {total_docs}")
        
        # Cerrar
        client.close()
        print("🎉 Recolección completada exitosamente")
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
