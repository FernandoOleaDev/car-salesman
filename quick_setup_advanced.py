#!/usr/bin/env python3
"""
CarBot Pro - Advanced Multi-Agent System Quick Setup
====================================================

Script de configuración rápida para el sistema multiagente avanzado de venta de coches.
Configura el entorno, instala dependencias y prepara el sistema para la demo.

Autor: Eduardo Hilario, CTO IA For Transport
Para: AI Agents Day Demo
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import json

def print_header():
    """Print setup header"""
    print("=" * 70)
    print("🚗 CarBot Pro - Advanced Multi-Agent System Setup")
    print("=" * 70)
    print("Demo para AI Agents Day")
    print("Autor: Eduardo Hilario, CTO IA For Transport")
    print("=" * 70)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Verificando versión de Python...")
    
    if sys.version_info < (3, 8):
        print("❌ Error: Se requiere Python 3.8 o superior")
        print(f"   Versión actual: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} - Compatible")
    return True

def create_virtual_environment():
    """Create and activate virtual environment"""
    print("\n🔧 Configurando entorno virtual...")
    
    venv_path = Path(".venv")
    
    if venv_path.exists():
        print("⚠️  Entorno virtual existente encontrado")
        response = input("¿Deseas recrearlo? (y/N): ").lower().strip()
        if response == 'y':
            print("🗑️  Eliminando entorno virtual existente...")
            shutil.rmtree(venv_path)
        else:
            print("✅ Usando entorno virtual existente")
            return True
    
    try:
        print("📦 Creando nuevo entorno virtual...")
        subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)
        print("✅ Entorno virtual creado exitosamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error creando entorno virtual: {e}")
        return False

def get_pip_command():
    """Get the correct pip command for the platform"""
    if os.name == 'nt':  # Windows
        return [".venv/Scripts/pip"]
    else:  # Unix/Linux/macOS
        return [".venv/bin/pip"]

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Instalando dependencias...")
    
    pip_cmd = get_pip_command()
    
    try:
        # Upgrade pip first
        print("🔄 Actualizando pip...")
        subprocess.run(pip_cmd + ["install", "--upgrade", "pip"], check=True)
        
        # Install requirements
        print("📥 Instalando dependencias del proyecto...")
        subprocess.run(pip_cmd + ["install", "-r", "requirements.txt"], check=True)
        
        print("✅ Dependencias instaladas exitosamente")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias: {e}")
        return False

def setup_environment_file():
    """Setup environment configuration file"""
    print("\n🔑 Configurando variables de entorno...")
    
    env_file = Path(".env")
    config_file = Path("config.env")
    
    if env_file.exists():
        print("⚠️  Archivo .env existente encontrado")
        response = input("¿Deseas sobrescribirlo? (y/N): ").lower().strip()
        if response != 'y':
            print("✅ Manteniendo configuración existente")
            return True
    
    if config_file.exists():
        print("📋 Copiando configuración desde config.env...")
        shutil.copy(config_file, env_file)
    else:
        print("📝 Creando archivo .env...")
        env_content = """# CarBot Pro - API Keys Configuration
# Añade tus claves reales aquí

# REQUERIDA: OpenAI API Key para los modelos de lenguaje
OPENAI_API_KEY=sk-your_openai_api_key_here

# OPCIONAL: SerpAPI Key para búsqueda web en tiempo real
SERPAPI_API_KEY=your_serpapi_key_here

# Configuración de la base de datos
INVENTORY_PATH=data/enhanced_inventory.csv

# Configuración del sistema
DEBUG_MODE=true
LOG_LEVEL=INFO
"""
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(env_content)
    
    print("✅ Archivo .env configurado")
    print("⚠️  IMPORTANTE: Edita el archivo .env con tus claves API reales")
    return True

def verify_data_files():
    """Verify that data files exist"""
    print("\n📊 Verificando archivos de datos...")
    
    data_dir = Path("data")
    enhanced_inventory = data_dir / "enhanced_inventory.csv"
    
    if not data_dir.exists():
        print("📁 Creando directorio de datos...")
        data_dir.mkdir()
    
    if enhanced_inventory.exists():
        print("✅ Inventario enriquecido encontrado")
        # Check file size
        file_size = enhanced_inventory.stat().st_size
        if file_size > 1000:  # At least 1KB
            print(f"✅ Archivo de inventario válido ({file_size} bytes)")
        else:
            print("⚠️  Archivo de inventario parece estar vacío")
    else:
        print("❌ Archivo de inventario enriquecido no encontrado")
        print("   Se requiere: data/enhanced_inventory.csv")
        return False
    
    return True

def verify_system_files():
    """Verify that all system files exist"""
    print("\n🔍 Verificando archivos del sistema...")
    
    required_files = [
        "enhanced_app.py",
        "advanced_multi_agent_system.py",
        "enhanced_inventory_manager.py",
        "requirements.txt"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"❌ Archivos faltantes: {', '.join(missing_files)}")
        return False
    
    print("✅ Todos los archivos del sistema están presentes")
    return True

def create_demo_script():
    """Create demo script file"""
    print("\n🎬 Creando guión de demo...")
    
    demo_script = {
        "demo_title": "CarBot Pro - Sistema Multiagente Avanzado",
        "presenter": "Eduardo Hilario, CTO IA For Transport",
        "duration": "30 minutos",
        "sections": {
            "1_demo": {
                "title": "Demostración en Vivo (8-10 min)",
                "prompts": [
                    {
                        "step": 1,
                        "role": "Cliente",
                        "prompt": "Hola, estoy buscando un coche",
                        "expected": "Saludo de Carlos y construcción de rapport"
                    },
                    {
                        "step": 2,
                        "role": "Cliente", 
                        "prompt": "Necesito un coche más grande y seguro porque hemos tenido un bebé",
                        "expected": "Carlos actualiza perfil y muestra comprensión"
                    },
                    {
                        "step": 3,
                        "role": "Cliente",
                        "prompt": "Quiero un sedan rojo que no tenga más de 2 años",
                        "expected": "Carlos consulta al manager y busca en inventario"
                    },
                    {
                        "step": 4,
                        "role": "Cliente",
                        "prompt": "Me interesan los BMW",
                        "expected": "Carlos refina búsqueda y presenta opciones"
                    },
                    {
                        "step": 5,
                        "role": "Cliente",
                        "prompt": "¿Qué características de seguridad tiene para bebés?",
                        "expected": "Carlos consulta a María para investigación"
                    },
                    {
                        "step": 6,
                        "role": "Cliente",
                        "prompt": "¿Qué espacio de maletero tiene el BMW X3?",
                        "expected": "María proporciona datos específicos"
                    },
                    {
                        "step": 7,
                        "role": "Cliente",
                        "prompt": "¿Cuál es el precio del BMW X3 negro?",
                        "expected": "Carlos consulta al manager para precio"
                    },
                    {
                        "step": 8,
                        "role": "Cliente",
                        "prompt": "¿Pueden hacer algún descuento?",
                        "expected": "Negociación entre Carlos y manager"
                    },
                    {
                        "step": 9,
                        "role": "Cliente",
                        "prompt": "Me lo quedo",
                        "expected": "Carlos finaliza venta y actualiza inventario"
                    }
                ]
            },
            "2_code_review": {
                "title": "Revisión de Código (20-22 min)",
                "topics": [
                    "Arquitectura multiagente",
                    "Gestión de inventario inteligente",
                    "Sistema de comunicación entre agentes",
                    "Logs y analytics en tiempo real",
                    "Integración con APIs externas",
                    "Manejo de estados y memoria"
                ]
            }
        },
        "key_features": [
            "Sistema multiagente con roles especializados",
            "Búsqueda inteligente en inventario enriquecido",
            "Investigación web en tiempo real",
            "Negociación automática entre agentes",
            "Perfilado dinámico de clientes",
            "Logs detallados y analytics",
            "Interfaz moderna con Streamlit"
        ]
    }
    
    with open("demo_script_advanced.json", 'w', encoding='utf-8') as f:
        json.dump(demo_script, f, indent=2, ensure_ascii=False)
    
    print("✅ Guión de demo creado: demo_script_advanced.json")

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "=" * 70)
    print("🎉 ¡CONFIGURACIÓN COMPLETADA EXITOSAMENTE!")
    print("=" * 70)
    print()
    print("📋 PRÓXIMOS PASOS:")
    print()
    print("1. 🔑 CONFIGURAR API KEYS:")
    print("   - Edita el archivo .env")
    print("   - Añade tu OpenAI API Key (REQUERIDA)")
    print("   - Añade tu SerpAPI Key (OPCIONAL)")
    print()
    print("2. 🚀 EJECUTAR LA APLICACIÓN:")
    
    if os.name == 'nt':  # Windows
        print("   .venv\\Scripts\\activate")
        print("   streamlit run enhanced_app.py")
    else:  # Unix/Linux/macOS
        print("   source .venv/bin/activate")
        print("   streamlit run enhanced_app.py")
    
    print()
    print("3. 🎬 PREPARAR DEMO:")
    print("   - Revisa demo_script_advanced.json")
    print("   - Practica los prompts sugeridos")
    print("   - Verifica que todos los agentes respondan")
    print()
    print("4. 🔧 MODO DEBUG:")
    print("   - Activa logs detallados en la interfaz")
    print("   - Monitorea comunicaciones entre agentes")
    print("   - Verifica analytics en tiempo real")
    print()
    print("=" * 70)
    print("🎯 ¡LISTO PARA LA DEMO DE AI AGENTS DAY!")
    print("=" * 70)

def main():
    """Main setup function"""
    print_header()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Setup environment file
    if not setup_environment_file():
        sys.exit(1)
    
    # Verify data files
    if not verify_data_files():
        sys.exit(1)
    
    # Verify system files
    if not verify_system_files():
        sys.exit(1)
    
    # Create demo script
    create_demo_script()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main() 