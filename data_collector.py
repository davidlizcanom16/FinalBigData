import requests
import json
import random
from datetime import datetime
import time

class EcommerceDataCollector:
    """
    Colector de datos de e-commerce con simulación de cambios en tiempo real
    """
    
    def __init__(self):
        self.api_url = "https://fakestoreapi.com/products"
        self.productos_base = []
        
    def obtener_productos_base(self):
    """Obtiene los productos base de la API"""
    print("📡 Obteniendo productos de Fake Store API...")
    try:
        # Headers para evitar bloqueos
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(self.api_url, headers=headers, timeout=10)
        if response.status_code == 200:  # <-- AQUÍ estaba el error (tenía un espacio extra)
            self.productos_base = response.json()
            print(f"✅ {len(self.productos_base)} productos obtenidos exitosamente")
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return False
    
    def simular_variacion_precio(self, precio_original):
        """
        Simula variación de precio realista (±10%)
        """
        variacion = random.uniform(-0.10, 0.10)  # Variación de -10% a +10%
        nuevo_precio = precio_original * (1 + variacion)
        return round(nuevo_precio, 2)
    
    def generar_metricas_tiempo_real(self):
        """
        Genera métricas adicionales que cambiarían en tiempo real
        """
        return {
            'stock': random.randint(0, 500),
            'ventas_dia': random.randint(0, 100),
            'views_hora': random.randint(50, 5000),
            'en_carrito': random.randint(0, 50),
            'disponible': random.choice([True, True, True, False])  # 75% disponible
        }
    
    def enriquecer_producto(self, producto):
        """
        Toma un producto base y le agrega datos simulados en tiempo real
        """
        producto_enriquecido = producto.copy()
        
        # Agregar timestamp actual
        producto_enriquecido['timestamp'] = datetime.now().isoformat()
        
        # Guardar precio original
        producto_enriquecido['precio_original'] = producto['price']
        
        # Simular nuevo precio
        producto_enriquecido['precio_actual'] = self.simular_variacion_precio(producto['price'])
        
        # Calcular cambio porcentual
        cambio = ((producto_enriquecido['precio_actual'] - producto['price']) / producto['price']) * 100
        producto_enriquecido['cambio_precio_porcentaje'] = round(cambio, 2)
        
        # Agregar métricas en tiempo real
        metricas = self.generar_metricas_tiempo_real()
        producto_enriquecido.update(metricas)
        
        # Agregar descuento aleatorio
        producto_enriquecido['descuento'] = random.choice([0, 0, 0, 5, 10, 15, 20])
        
        return producto_enriquecido
    
    def recolectar_datos(self, num_iteraciones=1):
        """
        Recolecta datos múltiples veces para simular cambios en el tiempo
        """
        if not self.productos_base:
            if not self.obtener_productos_base():
                return None
        
        print(f"\n🔄 Generando {num_iteraciones} snapshot(s) de datos...")
        
        todos_los_snapshots = []
        
        for i in range(num_iteraciones):
            print(f"\n📸 Snapshot #{i+1} - {datetime.now().strftime('%H:%M:%S')}")
            
            snapshot = {
                'snapshot_id': i + 1,
                'timestamp': datetime.now().isoformat(),
                'productos': []
            }
            
            # Enriquecer cada producto
            for producto in self.productos_base:
                producto_enriquecido = self.enriquecer_producto(producto)
                snapshot['productos'].append(producto_enriquecido)
            
            todos_los_snapshots.append(snapshot)
            
            # Mostrar resumen
            precios = [p['precio_actual'] for p in snapshot['productos']]
            print(f"  💰 Precio promedio: ${sum(precios)/len(precios):.2f}")
            print(f"  📦 Productos en stock: {sum(1 for p in snapshot['productos'] if p['disponible'])}/{len(snapshot['productos'])}")
            
            # Esperar antes del siguiente snapshot (si hay más)
            if i < num_iteraciones - 1:
                print(f"  ⏳ Esperando 3 segundos...")
                time.sleep(3)
        
        return todos_los_snapshots
    
    def guardar_datos_local(self, datos, nombre_archivo='datos_ecommerce.json'):
        """
        Guarda los datos en un archivo JSON local
        """
        try:
            with open(nombre_archivo, 'w', encoding='utf-8') as f:
                json.dump(datos, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Datos guardados en: {nombre_archivo}")
            return True
        except Exception as e:
            print(f"\n❌ Error al guardar: {e}")
            return False
    
    def mostrar_resumen(self, datos):
        """
        Muestra un resumen de los datos recolectados
        """
        print("\n" + "="*60)
        print("📊 RESUMEN DE DATOS RECOLECTADOS")
        print("="*60)
        
        total_snapshots = len(datos)
        total_productos = len(datos[0]['productos']) if datos else 0
        
        print(f"\n📸 Total de snapshots: {total_snapshots}")
        print(f"🛍️  Productos por snapshot: {total_productos}")
        print(f"📦 Total de registros: {total_snapshots * total_productos}")
        
        if datos:
            # Análisis de primer snapshot
            primer_snapshot = datos[0]['productos']
            
            categorias = {}
            for p in primer_snapshot:
                cat = p['category']
                if cat not in categorias:
                    categorias[cat] = 0
                categorias[cat] += 1
            
            print(f"\n📑 Distribución por categoría:")
            for cat, count in categorias.items():
                print(f"  - {cat}: {count} productos")
            
            # Rango de precios
            precios = [p['precio_actual'] for p in primer_snapshot]
            print(f"\n💰 Rango de precios:")
            print(f"  - Mínimo: ${min(precios):.2f}")
            print(f"  - Máximo: ${max(precios):.2f}")
            print(f"  - Promedio: ${sum(precios)/len(precios):.2f}")
        
        print("\n" + "="*60)


# ============================================
# EJECUCIÓN DEL SCRIPT
# ============================================

if __name__ == "__main__":
    print("="*60)
    print("🚀 ECOMMERCE DATA COLLECTOR - SIMULACIÓN TIEMPO REAL")
    print("="*60)
    
    # Crear instancia del colector
    collector = EcommerceDataCollector()
    
    # Preguntar cuántos snapshots generar
    print("\n¿Cuántos snapshots quieres generar?")
    print("(Cada snapshot simula datos en un momento diferente)")
    print("Recomendado: 3-5 para pruebas")
    
    try:
        num_snapshots = int(input("\nNúmero de snapshots: ") or "3")
    except:
        num_snapshots = 3
        print(f"Usando valor por defecto: {num_snapshots}")
    
    # Recolectar datos
    datos = collector.recolectar_datos(num_snapshots)
    
    if datos:
        # Mostrar resumen
        collector.mostrar_resumen(datos)
        
        # Guardar datos localmente
        collector.guardar_datos_local(datos)
        
        print("\n✅ ¡PROCESO COMPLETADO!")
        print("\n📋 Próximos pasos:")
        print("  1. Revisar el archivo 'datos_ecommerce.json'")
        print("  2. Configurar Azure Cosmos DB")
        print("  3. Subir estos datos a la nube")
    else:
        print("\n❌ No se pudieron recolectar datos")
