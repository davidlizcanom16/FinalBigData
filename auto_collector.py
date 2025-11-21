import time
import schedule
from datetime import datetime
from data_collector import EcommerceDataCollector
from azure_connector import AzureCosmosConnector

class AutomaticCollector:
    """
    Recolector automático que obtiene datos cada minuto
    """
    
    def __init__(self, intervalo_minutos=1):
        self.intervalo = intervalo_minutos
        self.data_collector = EcommerceDataCollector()
        self.azure_connector = AzureCosmosConnector()
        self.contador_recolecciones = 0
        
        # Conectar a Azure al iniciar
        if not self.azure_connector.conectar():
            raise Exception("No se pudo conectar a Azure")
    
    def recolectar_y_subir(self):
        """Recolecta datos y los sube a Azure"""
        self.contador_recolecciones += 1
        
        print("\n" + "="*70)
        print(f"🔄 RECOLECCIÓN AUTOMÁTICA #{self.contador_recolecciones}")
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        try:
            # Recolectar un snapshot
            datos = self.data_collector.recolectar_datos(num_iteraciones=1)
            
            if datos and len(datos) > 0:
                snapshot = datos[0]
                productos = snapshot['productos']
                
                # Subir cada producto a Azure
                insertados = 0
                for producto in productos:
                    # Agregar metadata
                    producto['snapshot_id'] = self.contador_recolecciones
                    producto['snapshot_timestamp'] = snapshot['timestamp']
                    
                    # Insertar en Azure
                    self.azure_connector.collection.insert_one(producto)
                    insertados += 1
                
                print(f"✅ {insertados} productos insertados en Azure")
                print(f"📊 Total acumulado: {self.contador_recolecciones * 20} documentos")
                
            else:
                print("❌ No se pudieron recolectar datos")
                
        except Exception as e:
            print(f"❌ Error durante la recolección: {e}")
    
    def iniciar(self, num_recolecciones=None):
        """
        Inicia la recolección automática
        
        Args:
            num_recolecciones: Número de recolecciones a hacer (None = infinito)
        """
        print("="*70)
        print("🚀 RECOLECTOR AUTOMÁTICO DE DATOS")
        print("="*70)
        print(f"⏱️  Intervalo: cada {self.intervalo} minuto(s)")
        
        if num_recolecciones:
            print(f"📊 Recolecciones programadas: {num_recolecciones}")
        else:
            print(f"📊 Recolecciones: continuas (presiona Ctrl+C para detener)")
        
        print("="*70)
        
        # Obtener productos base una vez
        print("\n📡 Obteniendo productos base de la API...")
        if not self.data_collector.obtener_productos_base():
            print("❌ Error al obtener productos base")
            return
        
        # Programar la tarea
        schedule.every(self.intervalo).minutes.do(self.recolectar_y_subir)
        
        # Hacer la primera recolección inmediatamente
        print("\n🎬 Iniciando primera recolección...")
        self.recolectar_y_subir()
        
        # Continuar con el schedule
        try:
            recolecciones_hechas = 1
            
            while True:
                # Si hay límite de recolecciones, verificar
                if num_recolecciones and recolecciones_hechas >= num_recolecciones:
                    print("\n✅ Número de recolecciones completado")
                    break
                
                # Ejecutar tareas programadas
                schedule.run_pending()
                
                # Verificar si se ejecutó una nueva recolección
                if self.contador_recolecciones > recolecciones_hechas:
                    recolecciones_hechas = self.contador_recolecciones
                
                # Esperar un segundo
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Recolección detenida por el usuario")
        
        finally:
            # Mostrar resumen final
            self.mostrar_resumen_final()
            self.azure_connector.cerrar_conexion()
    
    def mostrar_resumen_final(self):
        """Muestra resumen de la sesión"""
        print("\n" + "="*70)
        print("📊 RESUMEN DE LA SESIÓN")
        print("="*70)
        print(f"✅ Recolecciones completadas: {self.contador_recolecciones}")
        print(f"📦 Total de documentos insertados: {self.contador_recolecciones * 20}")
        
        # Obtener estadísticas de Azure
        try:
            total_docs = self.azure_connector.collection.count_documents({})
            print(f"☁️  Total en Azure Cosmos DB: {total_docs} documentos")
        except:
            pass
        
        print("="*70)


# ============================================
# EJECUCIÓN DEL SCRIPT
# ============================================

if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════╗
║        🤖 RECOLECTOR AUTOMÁTICO DE DATOS - AZURE COSMOS DB        ║
╚════════════════════════════════════════════════════════════════════╝
    """)
    
    print("Opciones:")
    print("  1. Recolección continua (cada 1 minuto hasta que detengas)")
    print("  2. Número específico de recolecciones (ej: 5 recolecciones)")
    print()
    
    try:
        opcion = input("Elige opción (1 o 2): ").strip()
        
        if opcion == "1":
            # Recolección continua
            collector = AutomaticCollector(intervalo_minutos=1)
            print("\n💡 Presiona Ctrl+C cuando quieras detener")
            time.sleep(2)
            collector.iniciar()
            
        elif opcion == "2":
            # Número específico
            num = int(input("¿Cuántas recolecciones? (recomendado: 5-10): "))
            collector = AutomaticCollector(intervalo_minutos=1)
            collector.iniciar(num_recolecciones=num)
            
        else:
            print("❌ Opción inválida")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")