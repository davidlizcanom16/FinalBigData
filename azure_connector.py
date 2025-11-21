import pymongo
import json
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

class AzureCosmosConnector:
    """Conector para Azure Cosmos DB con MongoDB API"""
    
    def __init__(self):
        self.connection_string = None
        
        # Primero intentar Streamlit
        try:
            import streamlit as st
            self.connection_string = st.secrets.get("COSMOS_CONNECTION_STRING")
        except:
            pass
        
        # Luego .env
        if not self.connection_string:
            self.connection_string = os.getenv('COSMOS_CONNECTION_STRING')
        
        self.client = None
        self.db = None
        self.collection = None
        self.database_name = "ecommerce_db"
        self.collection_name = "productos"
    
    def conectar(self):
        """Establece conexión con Azure Cosmos DB"""
        try:
            print("🔌 Conectando a Azure Cosmos DB...")
            self.client = pymongo.MongoClient(self.connection_string)
            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_name]
            self.client.server_info()
            print("✅ Conexión exitosa a Azure Cosmos DB")
            print(f"📊 Base de datos: {self.database_name}")
            print(f"📦 Colección: {self.collection_name}")
            return True
        except Exception as e:
            print(f"❌ Error al conectar: {e}")
            return False
    
    def subir_datos_desde_json(self, archivo_json='datos_ecommerce.json'):
        """Lee el archivo JSON local y sube los datos a Azure"""
        try:
            print(f"\n📂 Leyendo archivo: {archivo_json}")
            with open(archivo_json, 'r', encoding='utf-8') as f:
                datos = json.load(f)
            
            print(f"✅ Archivo leído: {len(datos)} snapshots encontrados")
            total_insertados = 0
            
            for snapshot in datos:
                snapshot_id = snapshot['snapshot_id']
                timestamp = snapshot['timestamp']
                productos = snapshot['productos']
                
                print(f"\n📸 Procesando Snapshot #{snapshot_id}...")
                print(f"   Timestamp: {timestamp}")
                print(f"   Productos: {len(productos)}")
                
                for producto in productos:
                    producto['snapshot_id'] = snapshot_id
                    producto['snapshot_timestamp'] = timestamp
                    resultado = self.collection.insert_one(producto)
                    total_insertados += 1
                
                print(f"   ✅ {len(productos)} productos insertados")
            
            print(f"\n🎉 TOTAL: {total_insertados} documentos insertados en Azure")
            return total_insertados
        except FileNotFoundError:
            print(f"❌ Error: No se encontró el archivo {archivo_json}")
            return 0
        except Exception as e:
            print(f"❌ Error al subir datos: {e}")
            return 0
    
    def obtener_estadisticas(self):
        """Muestra estadísticas de los datos en Azure"""
        try:
            print("\n" + "="*60)
            print("📊 ESTADÍSTICAS DE DATOS EN AZURE COSMOS DB")
            print("="*60)
            
            total_docs = self.collection.count_documents({})
            print(f"\n📦 Total de documentos: {total_docs}")
            
            pipeline_snapshots = [
                {"$group": {"_id": "$snapshot_id", "count": {"$sum": 1}}}
            ]
            snapshots = list(self.collection.aggregate(pipeline_snapshots))
            print(f"\n📸 Snapshots almacenados: {len(snapshots)}")
            for snap in sorted(snapshots, key=lambda x: x['_id']):
                print(f"   - Snapshot #{snap['_id']}: {snap['count']} productos")
            
            pipeline_categorias = [
                {"$group": {"_id": "$category", "count": {"$sum": 1}}}
            ]
            categorias = list(self.collection.aggregate(pipeline_categorias))
            print(f"\n📑 Distribución por categoría:")
            for cat in categorias:
                print(f"   - {cat['_id']}: {cat['count']} documentos")
            
            pipeline_precios = [
                {"$group": {
                    "_id": None,
                    "precio_min": {"$min": "$precio_actual"},
                    "precio_max": {"$max": "$precio_actual"},
                    "precio_promedio": {"$avg": "$precio_actual"}
                }}
            ]
            precios = list(self.collection.aggregate(pipeline_precios))
            if precios:
                p = precios[0]
                print(f"\n💰 Rango de precios actuales:")
                print(f"   - Mínimo: ${p['precio_min']:.2f}")
                print(f"   - Máximo: ${p['precio_max']:.2f}")
                print(f"   - Promedio: ${p['precio_promedio']:.2f}")
            
            print("\n" + "="*60)
        except Exception as e:
            print(f"❌ Error al obtener estadísticas: {e}")
    
    def obtener_muestra_datos(self, limite=3):
        """Muestra algunos documentos de ejemplo"""
        try:
            print(f"\n📋 MUESTRA DE DATOS (primeros {limite} documentos):")
            print("-"*60)
            
            documentos = self.collection.find().limit(limite)
            
            for i, doc in enumerate(documentos, 1):
                print(f"\n🛍️  Documento #{i}:")
                print(f"   ID: {doc.get('id')}")
                print(f"   Título: {doc.get('title')[:50]}...")
                print(f"   Precio original: ${doc.get('precio_original')}")
                print(f"   Precio actual: ${doc.get('precio_actual')}")
                print(f"   Stock: {doc.get('stock')}")
                print(f"   Ventas/día: {doc.get('ventas_dia')}")
                print(f"   Snapshot: #{doc.get('snapshot_id')}")
            
            print("-"*60)
        except Exception as e:
            print(f"❌ Error al obtener muestra: {e}")
    
    def limpiar_coleccion(self):
        """Elimina todos los documentos de la colección"""
        try:
            respuesta = input("\n⚠️  ¿Estás seguro de eliminar TODOS los datos? (escribe 'SI' para confirmar): ")
            if respuesta.upper() == 'SI':
                resultado = self.collection.delete_many({})
                print(f"🗑️  {resultado.deleted_count} documentos eliminados")
            else:
                print("❌ Operación cancelada")
        except Exception as e:
            print(f"❌ Error al limpiar: {e}")
    
    def cerrar_conexion(self):
        """Cierra la conexión con Azure"""
        if self.client:
            self.client.close()
            print("\n🔌 Conexión cerrada")


if __name__ == "__main__":
    print("="*60)
    print("☁️  AZURE COSMOS DB - SUBIR DATOS")
    print("="*60)
    
    connector = AzureCosmosConnector()
    
    if connector.conectar():
        print("\n" + "="*60)
        print("📤 SUBIENDO DATOS A AZURE")
        print("="*60)
        
        total = connector.subir_datos_desde_json('datos_ecommerce.json')
        
        if total > 0:
            connector.obtener_estadisticas()
            connector.obtener_muestra_datos(3)
            
            print("\n✅ ¡DATOS SUBIDOS EXITOSAMENTE A AZURE COSMOS DB!")
            print("\n💡 Próximos pasos:")
            print("   1. Verificar datos en Azure Portal")
            print("   2. Crear modelos de Machine Learning")
            print("   3. Construir el dashboard interactivo")
        
        connector.cerrar_conexion()
    else:
        print("\n❌ No se pudo establecer conexión con Azure")
