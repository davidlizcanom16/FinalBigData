import requests
import json
from datetime import datetime

# URL de la API de Fake Store
API_URL = "https://fakestoreapi.com/products"

print("=" * 50)
print("PROBANDO FAKE STORE API")
print("=" * 50)
print()

# Hacer la petición a la API
print("📡 Obteniendo datos de la API...")
response = requests.get(API_URL)

# Verificar si la petición fue exitosa
if response.status_code == 200:
    print("✅ ¡Conexión exitosa!")
    print()
    
    # Convertir la respuesta a JSON
    productos = response.json()
    
    # Mostrar información general
    print(f"📦 Total de productos obtenidos: {len(productos)}")
    print()
    
    # Mostrar los primeros 3 productos como ejemplo
    print("🛍️  PRIMEROS 3 PRODUCTOS:")
    print("-" * 50)
    
    for i, producto in enumerate(productos[:3], 1):
        print(f"\nProducto #{i}:")
        print(f"  ID: {producto['id']}")
        print(f"  Título: {producto['title']}")
        print(f"  Precio: ${producto['price']}")
        print(f"  Categoría: {producto['category']}")
        print(f"  Rating: {producto['rating']['rate']} ⭐ ({producto['rating']['count']} reviews)")
    
    print()
    print("-" * 50)
    
    # Mostrar todas las categorías disponibles
    categorias = set([p['category'] for p in productos])
    print(f"\n📑 CATEGORÍAS DISPONIBLES ({len(categorias)}):")
    for categoria in categorias:
        count = len([p for p in productos if p['category'] == categoria])
        print(f"  - {categoria}: {count} productos")
    
    print()
    print("=" * 50)
    print("✅ PRUEBA COMPLETADA EXITOSAMENTE")
    print("=" * 50)
    
else:
    print(f"❌ Error al conectar con la API. Código: {response.status_code}")