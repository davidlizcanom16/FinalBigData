#!/usr/bin/env python3
"""
Script para GitHub Actions - Recolecta y sube datos a Azure
"""

from data_collector import EcommerceDataCollector
from azure_connector import AzureCosmosConnector
import sys
import os

def main():
    try:
        print("🚀 Iniciando recolección de datos...")
        
        # Verificar que existe la connection string
        if not os.getenv('COSMOS_CONNECTION_STRING'):
            print("❌ Error: COSMOS_CONNECTION_STRING no encontrado")
            sys.exit(1)
        
        # Crear instancias
        collector = EcommerceDataCollector()
        azure = AzureCosmosConnector()
        
        # Conectar a Azure
        if not azure.conectar():
            print("❌ No se pudo conectar a Azure")
            sys.exit(1)
        
        print("✅ Conectado a Azure Cosmos DB")
        
        # Recolectar datos (1 snapshot = 20 productos)
        datos = collector.recolectar_datos(num_iteraciones=1)
        
        if not datos or len(datos) == 0:
            print("❌ No se pudieron recolectar datos de la API")
            azure.cerrar_conexion()
            sys.exit(1)
        
        # Obtener el snapshot
        snapshot = datos[0]
        productos = snapshot['productos']
        timestamp = snapshot['timestamp']
        
        print(f"📦 Recolectados {len(productos)} productos")
        
        # Subir a Azure
        insertados = 0
        for producto in productos:
            producto['snapshot_timestamp'] = timestamp
            producto['github_action'] = True  # Marcar como agregado por GitHub
            azure.collection.insert_one(producto)
            insertados += 1
        
        print(f"✅ {insertados} productos insertados en Azure")
        
        # Cerrar conexión
        azure.cerrar_conexion()
        
        print("🎉 Recolección completada exitosamente")
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
