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
        
        try:
            import streamlit as st
            if "COSMOS_CONNECTION_STRING" in st.secrets:
                self.connection_string = st.secrets["COSMOS_CONNECTION_STRING"]
                print("🔐 Usando Streamlit secrets")
        except ImportError:
            pass
        except Exception as e:
            print(f"⚠️ Error leyendo secrets: {e}")
        
        if not self.connection_string:
            self.connection_string = os.getenv('COSMOS_CONNECTION_STRING')
            if self.connection_string:
                print("🔐 Usando .env file")
        
        if not self.connection_string:
            print("❌ COSMOS_CONNECTION_STRING no encontrado")
        
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
            return True
        except Exception as e:
            print(f"❌ Error al conectar: {e}")
            return False
    
    def cerrar_conexion(self):
        """Cierra la conexión con Azure"""
        if self.client:
            self.client.close()
            print("\n🔌 Conexión cerrada")
