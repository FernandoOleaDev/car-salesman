"""
Sistema Avanzado Multi-Agente para Ventas de Coches
==================================================

Este módulo implementa un sistema de ventas de coches basado en inteligencia artificial
que utiliza múltiples agentes especializados trabajando en conjunto:
- Carlos: Agente de ventas principal (GPT-4o)
- María: Especialista en investigación (o4-mini)
- Manager: Coordinador de negocio y políticas

El sistema maneja todo el proceso de venta desde el saludo inicial hasta el cierre,
incluyendo búsqueda de inventario, investigación de vehículos, y gestión de objeciones.
"""

# ========================================
# IMPORTACIONES ESTÁNDAR DE PYTHON
# ========================================
import os          # Para operaciones del sistema operativo
import json        # Para manejo de datos JSON
import logging     # Para registro de eventos y debugging
from typing import List, Dict, Any, Optional  # Para tipado estático
from datetime import datetime                 # Para manejo de fechas y tiempos
from dataclasses import dataclass, asdict    # Para estructuras de datos
from enum import Enum                         # Para enumeraciones

# ========================================
# IMPORTACIONES DE LANGCHAIN (IA/LLM)
# ========================================
from langchain.agents import Tool, AgentExecutor, create_react_agent  # Agentes y herramientas
from langchain.prompts import PromptTemplate                          # Plantillas de prompts
from langchain_openai import ChatOpenAI                              # Modelo OpenAI
from langchain_community.utilities import SerpAPIWrapper              # Búsqueda web
from langchain_core.messages import HumanMessage, AIMessage          # Tipos de mensajes
from langchain.memory import ConversationBufferWindowMemory          # Memoria conversacional

# ========================================
# IMPORTACIONES LOCALES
# ========================================
from enhanced_inventory_manager import get_inventory_manager, CarSearchResult  # Gestor de inventario
from dotenv import load_dotenv  # Para cargar variables de entorno

# Cargar variables de entorno desde archivo .env (API keys, configuraciones)
load_dotenv()

# ========================================
# CONFIGURACIÓN DEL SISTEMA DE LOGGING
# ========================================
# Configurar el sistema de registro para monitorear actividades y errores
logging.basicConfig(
    level=logging.INFO,  # Nivel de logging: INFO captura información general
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Formato: timestamp - nombre - nivel - mensaje
    handlers=[
        logging.FileHandler('carbot_system.log'),  # Guardar logs en archivo para persistencia
        logging.StreamHandler()                    # También mostrar logs en la consola
    ]
)
logger = logging.getLogger(__name__)  # Crear logger específico para este módulo

# ========================================
# DEFINICIÓN DE ENUMERACIONES (ESTADOS Y ROLES)
# ========================================

class SalesStage(Enum):
    """
    Enumeración que define las etapas del proceso de venta de coches.
    Cada etapa representa una fase específica en el embudo de ventas.
    """
    GREETING = "greeting"                    # Saludo inicial y construcción de rapport
    DISCOVERY = "discovery"                  # Descubrimiento de necesidades del cliente
    PRESENTATION = "presentation"            # Presentación de vehículos específicos
    OBJECTION_HANDLING = "objection_handling"  # Manejo de objeciones y preocupaciones
    NEGOTIATION = "negotiation"              # Negociación de precios y términos
    CLOSING = "closing"                      # Cierre de la venta
    FOLLOW_UP = "follow_up"                 # Seguimiento post-venta

class AgentRole(Enum):
    """
    Enumeración que define los diferentes roles de agentes en el sistema.
    Cada agente tiene responsabilidades y características específicas.
    """
    CARLOS_SALES = "carlos_sales"            # Carlos - Agente principal de ventas
    MARIA_RESEARCH = "maria_research"        # María - Especialista en investigación
    MANAGER_COORDINATOR = "manager_coordinator"  # Manager - Coordinador de políticas y negocio

# ========================================
# ESTRUCTURAS DE DATOS (DATACLASSES)
# ========================================

@dataclass
class CustomerProfile:
    """
    Perfil completo del cliente que almacena toda la información relevante
    para personalizar la experiencia de venta y las recomendaciones.
    
    Esta clase encapsula tanto información demográfica como preferencias,
    necesidades, historial de interacciones y objeciones del cliente.
    """
    # Información básica del cliente
    name: Optional[str] = None                      # Nombre del cliente
    
    # Información financiera
    budget_min: Optional[int] = None                # Presupuesto mínimo en euros
    budget_max: Optional[int] = None                # Presupuesto máximo en euros
    
    # Preferencias de vehículo
    preferred_make: Optional[str] = None            # Marca preferida (BMW, Mercedes, etc.)
    preferred_color: Optional[str] = None           # Color preferido
    body_style_preference: Optional[str] = None     # Tipo de carrocería (SUV, sedán, etc.)
    fuel_type_preference: Optional[str] = None      # Tipo de combustible (gasolina, híbrido, eléctrico)
    
    # Información del contexto familiar/personal
    family_size: Optional[int] = None               # Número de miembros de la familia
    primary_use: Optional[str] = None               # Uso principal (trabajo, familiar, etc.)
    
    # Prioridades y preferencias especiales
    safety_priority: bool = False                   # Si la seguridad es una prioridad alta
    luxury_preference: bool = False                 # Si prefiere características de lujo
    eco_friendly: bool = False                      # Si busca opciones ecológicas
    
    # Listas dinámicas de información
    needs: List[str] = None                        # Lista de necesidades específicas
    objections: List[str] = None                   # Lista de objeciones expresadas
    interaction_history: List[Dict] = None         # Historial completo de interacciones
    
    def __post_init__(self):
        """
        Método ejecutado automáticamente después de la inicialización.
        Inicializa las listas como listas vacías si son None para evitar errores.
        """
        if self.needs is None:
            self.needs = []
        if self.objections is None:
            self.objections = []
        if self.interaction_history is None:
            self.interaction_history = []

@dataclass
class AgentCommunication:
    """
    Estructura que define la comunicación entre agentes del sistema.
    Permite rastrear y registrar todas las interacciones internas entre
    Carlos, María y el Manager para análisis y debugging.
    """
    from_agent: AgentRole      # Agente que envía el mensaje
    to_agent: AgentRole        # Agente que recibe el mensaje
    message_type: str          # Tipo de mensaje (consulta, respuesta, etc.)
    content: str               # Contenido completo del mensaje
    timestamp: datetime        # Marca de tiempo precisa
    priority: str = "normal"   # Prioridad del mensaje (normal, high, urgent)
    requires_response: bool = False  # Si el mensaje requiere una respuesta

# ========================================
# CLASE PRINCIPAL DEL SISTEMA MULTI-AGENTE
# ========================================

class AdvancedCarSalesSystem:
    """
    Sistema avanzado de ventas de coches con múltiples agentes de IA especializados.
    
    Esta clase orquesta la interacción entre tres agentes principales:
    - Carlos: Agente de ventas principal que interactúa directamente con el cliente
    - María: Especialista en investigación que proporciona información detallada
    - Manager: Coordinador de políticas de negocio y decisiones estratégicas
    
    El sistema maneja todo el ciclo de venta desde el primer contacto hasta el cierre.
    """
    
    def __init__(self, openai_api_key: str, serpapi_api_key: str = None):
        """
        Inicializar el sistema multi-agente con las configuraciones necesarias.
        
        Args:
            openai_api_key (str): Clave API de OpenAI para los modelos de lenguaje
            serpapi_api_key (str, optional): Clave API de SerpAPI para búsquedas web
        """
        # Almacenar las claves API para uso posterior
        self.openai_api_key = openai_api_key
        self.serpapi_api_key = serpapi_api_key
        
        # Inicializar el gestor de inventario que maneja la base de datos de vehículos
        self.inventory_manager = get_inventory_manager()
        
        # Inicializar el perfil del cliente (comienza vacío)
        self.customer_profile = CustomerProfile()
        # Establecer la etapa inicial de venta (saludo)
        self.sales_stage = SalesStage.GREETING
        
        # Sistemas de comunicación y seguimiento
        self.agent_communications = []      # Lista de comunicaciones entre agentes
        self.conversation_log = []          # Registro completo de la conversación
        self.carlos_customer_notes: List[str] = []  # Notas personales de Carlos sobre el cliente
        
        # ========================================
        # CONFIGURACIÓN DE MODELOS DE LENGUAJE
        # ========================================
        
        # Carlos: Agente de ventas principal
        # Usa GPT-4o con temperatura alta para conversaciones creativas y persuasivas
        self.carlos_llm = ChatOpenAI(
            temperature=0.8,              # Alta creatividad para técnicas de venta
            openai_api_key=openai_api_key,
            model_name="gpt-4o",         # Modelo más avanzado para ventas complejas
            max_tokens=1000              # Respuestas detalladas
        )
        
        # María: Especialista en investigación
        # Usa o4-mini con temperatura baja para análisis factuales y objetivos
        self.maria_llm = ChatOpenAI(
            temperature=1,               # Baja temperatura para precisión factual
            openai_api_key=openai_api_key,
            model_name="o4-mini",        # Modelo especializado para análisis
            max_tokens=800               # Informes concisos pero completos
        )
        
        # Manager: Coordinador de políticas de negocio
        # Usa GPT-4o con temperatura balanceada para decisiones estratégicas
        self.manager_llm = ChatOpenAI(
            temperature=0.4,             # Temperatura balanceada para decisiones
            openai_api_key=openai_api_key,
            model_name="gpt-4o",         # Modelo avanzado para coordinación inteligente
            max_tokens=600               # Respuestas directas y estratégicas
        )
        
        # ========================================
        # SISTEMA DE MEMORIA CONVERSACIONAL
        # ========================================
        
        # Inicializar memoria para Carlos (mantiene contexto de las últimas 10 interacciones)
        self.carlos_memory = ConversationBufferWindowMemory(
            memory_key="chat_history",   # Clave para acceder al historial
            input_key="input",          # Clave para las entradas del usuario
            k=10,                       # Mantener últimas 10 interacciones
            return_messages=True        # Devolver mensajes estructurados
        )
        
        # ========================================
        # INICIALIZACIÓN DE HERRAMIENTAS Y AGENTES
        # ========================================
        
        # Crear herramientas avanzadas que los agentes pueden usar
        self.tools = self._create_advanced_tools()
        # Crear los agentes especializados
        self.carlos_agent = self._create_carlos_agent()
        self.maria_agent = self._create_maria_agent()
        self.manager_agent = self._create_manager_agent()
        
        # Registrar inicialización exitosa
        logger.info("🚀 Advanced Car Sales System initialized successfully")
    
    def _perform_intelligent_inventory_search(self, query: str) -> str:
        """
        Método auxiliar para realizar búsqueda inteligente en el inventario.
        Utilizado por el Manager para encontrar vehículos que coincidan con los criterios.
        
        Args:
            query (str): Consulta de búsqueda con criterios del cliente
            
        Returns:
            str: Resultados formateados de la búsqueda o mensaje de error
        """
        # No se registra directamente la acción del agente aquí, ya que es una capacidad interna del sistema
        # La acción del manager se registrará cuando decida usar esto
        try:
            # Realizar búsqueda en el inventario con máximo 8 resultados
            results_objects = self.inventory_manager.intelligent_search(query, max_results=8)
            # Formatear resultados para mostrar a los agentes
            formatted_results = self.inventory_manager.format_search_results_for_agent(results_objects, max_display=len(results_objects))
            
            # Registrar el resultado de la búsqueda
            logger.info(f"⚙️ System performed inventory search for query '{query}', found {len(results_objects)} vehicles.")
            return formatted_results
        except Exception as e:
            logger.error(f"❌ Error in _perform_intelligent_inventory_search: {e}")
            return "❌ Error interno al realizar la búsqueda de inventario."
    
    def _create_advanced_tools(self) -> List[Tool]:
        """
        Crear herramientas avanzadas para el sistema multi-agente.
        Estas herramientas permiten a Carlos interactuar con otros agentes y el sistema.
        
        Returns:
            List[Tool]: Lista de herramientas disponibles para los agentes
        """
        tools = []
        
        # ========================================
        # HERRAMIENTA DE CONSULTA AL MANAGER
        # ========================================
        def consult_manager(request: str) -> str:
            """
            Consultar con el manager para decisiones de precios, prioridades e inventario.
            Esta herramienta permite a Carlos obtener orientación estratégica.
            """
            # Registrar la solicitud de consulta
            self._log_agent_communication(
                AgentRole.CARLOS_SALES,
                AgentRole.MANAGER_COORDINATOR,
                "consultation_request",
                request
            )
            
            try:
                # Lógica de negocios y toma de decisiones del manager
                manager_response = self._manager_decision_engine(request)
                
                # Registrar la respuesta del manager
                self._log_agent_communication(
                    AgentRole.MANAGER_COORDINATOR,
                    AgentRole.CARLOS_SALES,
                    "consultation_response",
                    manager_response
                )
                
                return manager_response
                
            except Exception as e:
                logger.error(f"❌ Error consulting manager: {e}")
                return "El manager no está disponible en este momento. Procede con las políticas estándar."
        
        tools.append(Tool(
            name="ConsultManager",
            func=consult_manager,
            description="Consult with the sales manager for pricing decisions, inventory priorities, and business policies"
        ))
        
        # ========================================
        # HERRAMIENTA DE INVESTIGACIÓN VÍA MARÍA
        # ========================================
        def research_vehicle_info(query: str) -> str:
            """
            Investigar información detallada de vehículos, reseñas y datos de mercado.
            María proporciona análisis experto basado en búsquedas web y conocimiento.
            """
            # Registrar solicitud de investigación
            self._log_agent_communication(
                AgentRole.CARLOS_SALES,
                AgentRole.MARIA_RESEARCH,
                "research_request",
                query
            )
            
            try:
                # Motor de investigación de María
                research_result = self._maria_research_engine(query)
                
                # Registrar respuesta de investigación
                self._log_agent_communication(
                    AgentRole.MARIA_RESEARCH,
                    AgentRole.CARLOS_SALES,
                    "research_response",
                    research_result
                )
                
                return research_result
                
            except Exception as e:
                logger.error(f"❌ Error in research: {e}")
                return "No pude obtener información adicional en este momento."
        
        tools.append(Tool(
            name="ResearchVehicleInfo",
            func=research_vehicle_info,
            description="Research detailed vehicle specifications, reviews, safety ratings, and market information"
        ))
        
        # ========================================
        # HERRAMIENTA DE ACTUALIZACIÓN DE PERFIL DEL CLIENTE
        # ========================================
        def update_customer_profile(info: str) -> str:
            """
            Actualizar el perfil del cliente con nueva información obtenida en la conversación.
            Extrae automáticamente preferencias, necesidades, presupuesto, etc.
            """
            self._log_agent_action(AgentRole.CARLOS_SALES, "profile_update", info)
            
            try:
                # Procesar texto y extraer información estructurada
                self._update_customer_profile_from_text(info)
                profile_summary = self._get_customer_profile_summary()
                
                logger.info(f"📝 Customer profile updated: {profile_summary}")
                return f"Perfil actualizado: {profile_summary}"
                
            except Exception as e:
                logger.error(f"❌ Error updating customer profile: {e}")
                return "Error actualizando el perfil del cliente."
        
        tools.append(Tool(
            name="UpdateCustomerProfile",
            func=update_customer_profile,
            description="Update customer profile with preferences, needs, budget, and other relevant information"
        ))
        
        # ========================================
        # HERRAMIENTA DE GESTIÓN DE ETAPAS DE VENTA
        # ========================================
        def update_sales_stage(stage: str) -> str:
            """
            Actualizar la etapa actual del proceso de venta.
            Permite seguimiento del progreso a través del embudo de ventas.
            """
            try:
                # Validar y convertir la etapa
                new_stage = SalesStage(stage.lower())
                old_stage = self.sales_stage
                self.sales_stage = new_stage
                
                # Registrar transición de etapa
                self._log_agent_action(
                    AgentRole.CARLOS_SALES,
                    "stage_transition",
                    f"{old_stage.value} -> {new_stage.value}"
                )
                
                return f"Etapa de venta actualizada a: {new_stage.value}"
                
            except ValueError:
                return f"Etapa de venta inválida: {stage}"
        
        tools.append(Tool(
            name="UpdateSalesStage",
            func=update_sales_stage,
            description="Update the current sales stage (greeting, discovery, presentation, objection_handling, negotiation, closing)"
        ))

        # ========================================
        # HERRAMIENTA DE FINALIZACIÓN DE VENTA Y RESERVA
        # ========================================
        def finalize_sale_and_reserve_vehicle(vin: str) -> str:
            """
            Finaliza la venta de un vehículo y lo marca como reservado en el inventario.
            Usar SOLO cuando el cliente haya confirmado explícitamente que quiere comprar.
            
            Args:
                vin (str): Número de identificación del vehículo (VIN) a reservar
                
            Returns:
                str: Mensaje de confirmación o error si la reserva falló
            """
            # Registrar intento de finalización de venta
            self._log_agent_action(AgentRole.CARLOS_SALES, "finalize_sale_attempt", f"VIN: {vin}")
            try:
                # Intentar reservar el vehículo en el sistema de inventario
                success = self.inventory_manager.reserve_vehicle(vin)
                if success:
                    # Aquí se podrían activar otras acciones post-venta en un sistema real
                    logger.info(f"🎉 Sale finalized and vehicle {vin} reserved by Carlos.")
                    return f"¡Excelente! El vehículo con VIN {vin} ha sido reservado exitosamente. El proceso de compra ha concluido. Gracias!"
                else:
                    logger.warning(f"⚠️ Carlos failed to reserve vehicle {vin}. It might be already reserved or VIN is incorrect.")
                    return f"Hubo un problema al intentar reservar el vehículo {vin}. Por favor, verifica el VIN o el estado del vehículo. Podría ser que ya no esté disponible."
            except Exception as e:
                logger.error(f"❌ Error during finalize_sale_and_reserve_vehicle tool: {e}", exc_info=True)
                return f"Ocurrió un error técnico al intentar reservar el vehículo {vin}."

        tools.append(Tool(
            name="FinalizeSaleAndReserveVehicle",
            func=finalize_sale_and_reserve_vehicle,
            description="Use to finalize a sale and reserve a specific vehicle by its VIN when the customer agrees to purchase."
        ))

        # ========================================
        # HERRAMIENTA DE RESPUESTA DIRECTA AL CLIENTE
        # ========================================
        def respond_to_client(response: str) -> str:
            """
            Entrega tu mensaje directamente al cliente. Usar cuando estés listo para comunicar tu respuesta.
            IMPORTANTE: Después de usar esta herramienta, DEBES usar el formato 'Final Answer:' para concluir.
            
            Args:
                response (str): El mensaje completo que quieres enviar al cliente
                
            Returns:
                str: La respuesta que se envió al cliente
            """
            logger.info(f"🗣️ CARLOS TO CLIENT (via RespondToClient tool): {response[:100]}...")
            return response

        tools.append(Tool(
            name="RespondToClient",
            func=respond_to_client,
            description="Use this tool to provide your final answer or response directly to the customer. This action concludes your processing for the current customer input, and the observation returned will be the final answer."
        ))

        # ========================================
        # HERRAMIENTA DE NOTAS PERSONALES DE CARLOS
        # ========================================
        def update_customer_notes(note_to_add: str, mode: str = "append") -> str:
            """
            Añade o sobrescribe notas en el bloc personal de Carlos sobre el cliente.
            'append' para añadir nueva nota. 'overwrite' para reemplazar todas las notas.
            Para capturar detalles, matices o declaraciones específicas del cliente.
            
            Args:
                note_to_add (str): El texto de la nota a añadir o usar para sobrescribir
                mode (str): 'append' o 'overwrite'. Por defecto 'append'
                
            Returns:
                str: Mensaje de confirmación de la acción realizada
            """
            # Registrar intento de actualización de notas
            self._log_agent_action(AgentRole.CARLOS_SALES, "update_customer_notes_attempt", f"Mode: {mode}, Note: {note_to_add[:50]}...")
            
            if mode.lower() == "overwrite":
                # Sobrescribir todas las notas con la nueva
                self.carlos_customer_notes = [note_to_add]
                logger.info(f"📝 Carlos's customer notes OVERWRITTEN. Current notes: {len(self.carlos_customer_notes)}")
                return f"Notas sobrescritas. Nueva nota: '{note_to_add[:100]}...'"
            elif mode.lower() == "append":
                # Añadir nueva nota a las existentes
                self.carlos_customer_notes.append(note_to_add)
                logger.info(f"📝 Carlos's customer note APPENDED. Total notes: {len(self.carlos_customer_notes)}")
                return f"Nota añadida: '{note_to_add[:100]}...'. Total de notas: {len(self.carlos_customer_notes)}."
            else:
                return "Modo inválido. Usa 'append' o 'overwrite'."

        tools.append(Tool(
            name="UpdateCustomerNotes",
            func=update_customer_notes,
            description="Gestiona tus notas personales sobre el cliente. Útil para detalles cualitativos. Modos: 'append', 'overwrite'."
        ))
        
        return tools
    
    def _create_carlos_agent(self) -> AgentExecutor:
        """Create Carlos - the expert sales agent"""
        
        # This specific prompt structure is required for ReAct agents.
        # It needs: {tools}, {tool_names}, {input}, {agent_scratchpad}, and {chat_history}
        carlos_prompt = PromptTemplate.from_template("""
Eres Carlos, un vendedor de coches experto con 15 años de experiencia, potenciado por IA avanzada (GPT-4o). 
Tu MISIÓN es guiar al cliente a través del proceso de venta para encontrar su coche ideal y cerrar la venta.
Debes ser carismático, conocedor y genuinamente preocupado por las necesidades del cliente.

PERSONALIDAD Y ESTILO:
- Cálido, profesional y confiable.
- Escuchas activamente y haces preguntas inteligentes para descubrir necesidades.
- Usas técnicas de venta consultiva. Nunca seas pasivo, siempre guía la conversación.
- Construyes rapport genuino.
- Manejas objeciones con empatía y datos concretos.

PROCESO DE VENTA ESTRUCTURADO (usa la herramienta UpdateSalesStage para transicionar):
1. GREETING: Saludo inicial, construir rapport.
2. DISCOVERY: Entender profundamente necesidades, presupuesto, preferencias del cliente.
3. PRESENTATION: Mostrar vehículos del inventario que coincidan perfectamente (usar ConsultManager para obtener opciones de inventario).
4. OBJECTION_HANDLING: Abordar preocupaciones con empatía y soluciones.
5. NEGOTIATION: Trabajar hacia un acuerdo (usar ConsultManager para precios/descuentos y consultas de inventario complejas).
6. CLOSING: Finalizar la venta de manera natural.

HERRAMIENTAS DISPONIBLES (DEBES usar estas herramientas para interactuar con el sistema):
{tools}

DESCRIPCIÓN DE HERRAMIENTAS (REFERENCIA RÁPIDA):
{tool_names}

CONTEXTO ACTUAL:
Etapa de venta actual: {sales_stage}
Perfil del cliente (actualizado continuamente): {customer_profile_summary}
Comunicaciones internas recientes (Manager/Maria): {internal_communications_summary}
TUS NOTAS PERSONALES DEL CLIENTE (Usa UpdateCustomerNotes para gestionarlas):
{customer_notes_summary}

INSTRUCCIONES CRÍTICAS PARA RESPONDER (DEBES SEGUIR ESTE FORMATO):

Cuando necesites usar una herramienta para obtener información o realizar una acción interna:
Thought: [Tu razonamiento detallado sobre la situación, qué necesitas hacer, y qué herramienta usar.]
Action: [UNA de las herramientas: ConsultManager, ResearchVehicleInfo, UpdateCustomerProfile, UpdateSalesStage, RespondToClient, FinalizeSaleAndReserveVehicle]
Action Input: [La entrada para la herramienta.]
Observation: [Resultado de la acción, rellenado por el sistema.]
... (Puedes repetir este ciclo de Thought/Action/Action Input/Observation varias veces si es necesario)

Cuando estés listo para responder al cliente Y CONCLUIR TU TURNO (o si la venta se cierra con FinalizeSaleAndReserveVehicle):
Thought: [Tu razonamiento final. Has usado RespondToClient o FinalizeSaleAndReserveVehicle si era necesario. Ahora vas a concluir.]
Final Answer: [Tu respuesta final completa al cliente. Esta es la que el cliente verá. Si usaste RespondToClient, esta respuesta DEBE ser idéntica a la entrada que le diste. Si usaste FinalizeSaleAndReserveVehicle, la Observación será tu respuesta final.]

EJEMPLO DE UN CICLO COMPLETO CON RESPUESTA AL CLIENTE:
Thought: El cliente pregunta por un SUV seguro. Primero actualizo su perfil.
Action: UpdateCustomerProfile
Action Input: El cliente necesita un SUV y la seguridad es prioritaria.
Observation: Perfil actualizado: Prioridad: Seguridad; Necesidades: SUV
Thought: El cliente también mencionó que tiene dos perros grandes. Voy a añadir esto a mis notas personales para recordarlo al hablar de espacio.
Action: UpdateCustomerNotes
Action Input: {{"note_to_add": "Tiene dos perros grandes - importante para espacio maletero/interior y facilidad de limpieza.", "mode": "append"}}
Observation: Nota añadida: 'Tiene dos perros grandes - importante para espacio maletero/interior y facilidad de limpieza.'. Total de notas: 1.
Thought: Ahora necesito consultar al manager para obtener opciones de SUVs seguros.
Action: ConsultManager
Action Input: El cliente busca un SUV seguro para familia. Necesito opciones del inventario.
Observation: (Respuesta del Manager con opciones de SUVs: BMW X3 2023 VIN WBAXG9C50DD123457, Volvo XC60 2022 VIN YV1CZARH4N123458. Notas del Manager: Priorizar BMW por margen.)
Thought: El Manager me ha dado dos opciones: BMW X3 y Volvo XC60. El cliente prioriza seguridad y tiene familia. Ambos son SUVs seguros. El manager sugiere priorizar el BMW X3. Voy a analizar cuál se ajusta mejor y lo presentaré de forma proactiva.
Action: UpdateSalesStage
Action Input: presentation
Observation: Etapa de venta actualizada a: presentation
Thought: Voy a recomendar el BMW X3, destacando su seguridad y características familiares, y mencionaré el Volvo como alternativa. Usaré RespondToClient.
Action: RespondToClient
Action Input: ¡Hola! He consultado con nuestro gerente y, basándome en tu necesidad de un SUV seguro para la familia, te recomiendo especialmente el **BMW X3 2023**. Es un vehículo excepcional en seguridad, espacioso y con excelentes características para el día a día familiar. Como alternativa también muy segura tenemos el Volvo XC60 2022. ¿Te gustaría que profundicemos en los detalles del BMW X3 o prefieres comparar ambos?
Observation: ¡Hola! He consultado con nuestro gerente y, basándome en tu necesidad de un SUV seguro para la familia, te recomiendo especialmente el **BMW X3 2023**. Es un vehículo excepcional en seguridad, espacioso y con excelentes características para el día a día familiar. Como alternativa también muy segura tenemos el Volvo XC60 2022. ¿Te gustaría que profundicemos en los detalles del BMW X3 o prefieres comparar ambos?
Thought: Ya he enviado la respuesta proactiva al cliente usando RespondToClient. La observación confirma el mensaje. Ahora, para terminar mi turno, debo usar el formato 'Final Answer:' repitiendo exactamente el mensaje que envié.
Final Answer: ¡Hola! He consultado con nuestro gerente y, basándome en tu necesidad de un SUV seguro para la familia, te recomiendo especialmente el **BMW X3 2023**. Es un vehículo excepcional en seguridad, espacioso y con excelentes características para el día a día familiar. Como alternativa también muy segura tenemos el Volvo XC60 2022. ¿Te gustaría que profundicemos en los detalles del BMW X3 o prefieres comparar ambos?

EJEMPLO DE CIERRE DE VENTA:
Thought: El cliente ha confirmado que quiere comprar el BMW X3. Recuerdo que el manager me proporcionó el VIN: WBAXG9C50DD123457 para este coche. Es crucial usar este VIN exacto.
Action: UpdateSalesStage
Action Input: closing
Observation: Etapa de venta actualizada a: closing
Thought: Ahora voy a usar la herramienta para reservar el vehículo, asegurándome de usar el VIN correcto WBAXG9C50DD123457.
Action: FinalizeSaleAndReserveVehicle
Action Input: WBAXG9C50DD123457
Observation: ¡Excelente! El vehículo con VIN WBAXG9C50DD123457 ha sido reservado exitosamente. El proceso de compra ha concluido. Gracias!
Thought: La venta se ha completado y el vehículo está reservado. La observación es la respuesta final para el cliente.
Final Answer: ¡Excelente! El vehículo con VIN WBAXG9C50DD123457 ha sido reservado exitosamente. El proceso de compra ha concluido. Gracias!

MANEJO DE ERRORES AL RESERVAR:
Thought: El cliente quiere el coche. Intenté reservarlo con el VIN XXXXX pero falló: "Hubo un problema al intentar reservar el vehículo XXXXX. Por favor, verifica el VIN o el estado del vehículo."
Thought: Primero, ¿estoy seguro de que XXXXX es el VIN correcto para el [Modelo de Coche Específico]? Debo revisar mis notas/historial de la conversación, especialmente la información del inventario del Manager o la investigación de Maria. (Si no estoy seguro o no lo encuentro, debo consultar al Manager por el VIN correcto del [Modelo de Coche Específico] ANTES de reintentar la reserva)
(Si estoy seguro que el VIN XXXXX era correcto)
Thought: El VIN XXXXX parece correcto. Consultaré al manager sobre por qué falló la reserva para este VIN específico.
Action: ConsultManager
Action Input: El cliente quiere el [Modelo de Coche Específico] con VIN XXXXX. Intenté reservarlo pero falló. ¿Podrías verificar la disponibilidad y el estado exacto de este vehículo con VIN XXXXX?
Observation: (Respuesta del Manager, ej: "El vehículo con VIN XXXXX fue reservado por otro agente hace 5 minutos." o "El VIN XXXXX es correcto y debería estar disponible, intenta de nuevo. Ha habido un error temporal en el sistema de reservas.")
Thought: (Basado en la respuesta del manager, decido cómo proceder. Si ya no está disponible, informo al cliente y sugiero alternativas. Si fue un error temporal, intento reservar de nuevo con el VIN correcto.)

SIGUIENDO DIRECTIVAS DEL MANAGER:
Thought: He recibido una respuesta del Manager con una sección "DIRECTIVA DE VENTA". Debo seguir estas instrucciones. El Manager prioriza el [Vehículo A] y como alternativa el [Vehículo B].
Action: UpdateSalesStage (si es necesario, ej. a 'presentation')
Action Input: presentation
Observation: Etapa de venta actualizada.
Thought: Voy a presentar el [Vehículo A] al cliente, utilizando los puntos y estrategia sugeridos por el Manager.
Action: RespondToClient
Action Input: (Mensaje al cliente presentando el Vehículo A, siguiendo las directivas del Manager y destacando los puntos relevantes para el cliente).
Observation: (El mensaje que se envió).
Thought: He presentado la opción prioritaria del Manager. Ahora concluyo mi turno.
Final Answer: (El mismo mensaje que se envió).

NEGOCIACIÓN CON EL MANAGER:
Thought: La directiva del Manager es presentar el Coche X, pero el cliente ha expresado muy fuertemente que quiere un Coche Y, que no estaba en la directiva. Debo consultar al Manager.
Action: ConsultManager
Action Input: Manager, entiendo la directiva de priorizar el Coche X. Sin embargo, el cliente está muy interesado en un Coche Y debido a [razón del cliente]. ¿Tenemos algún Coche Y en inventario que podamos considerar, o alguna estrategia para reenfocar al cliente en el Coche X, o una alternativa que se parezca más al Coche Y?
Observation: (Respuesta del Manager, ej: "Entendido, Carlos. Si el cliente está firme en el Coche Y, y tenemos el [VIN del Coche Y] disponible, puedes presentarlo. No es nuestra prioridad de margen, pero la satisfacción del cliente es importante. Destaca [aspecto positivo del Coche Y].")
Thought: (Procedo según la nueva instrucción del Manager.)
Action: RespondToClient
Action Input: (Respuesta al cliente basada en la nueva directiva/opción del Manager)
Observation: (Mensaje enviado)
Thought: He actuado según la nueva directiva del manager.
Final Answer: (Mismo mensaje enviado)

TÉCNICAS DE NEGOCIACIÓN AVANZADA CON EL CLIENTE:
Thought: El cliente está pidiendo un descuento en el [Modelo de Coche]. El precio es [Precio].
Action: RespondToClient
Action Input: Entiendo que el presupuesto es una consideración importante. El precio de [Precio] para este [Modelo de Coche] refleja su excelente estado, bajo kilometraje, y el completo paquete de características de seguridad y tecnología que incluye, como [Mencionar 2-3 características clave valoradas por el cliente]. Además, todos nuestros vehículos pasan por una rigurosa inspección de 150 puntos y vienen con una garantía integral. En lugar de un descuento directo, ¿estarías interesado en explorar nuestras opciones de financiación flexibles que podrían hacer que el pago mensual sea más cómodo? O quizás podemos revisar si alguna de las características opcionales podría ajustarse para encajar mejor en tu presupuesto sin comprometer la calidad esencial que buscas.
Observation: (Mensaje enviado)
Thought: He reforzado el valor y ofrecido alternativas a un descuento directo.
Final Answer: (Mismo mensaje enviado)


IMPORTANTE:
- Siempre debes tener el VIN exacto del vehículo antes de usar `FinalizeSaleAndReserveVehicle`. Encuéntralo en las respuestas del `ConsultManager` o `ResearchVehicleInfo`.
- Si `FinalizeSaleAndReserveVehicle` falla, no asumas inmediatamente que el coche no está disponible. Primero verifica que usaste el VIN correcto. Si el VIN era correcto, consulta al `ConsultManager` sobre el problema con ESE VIN específico.
- Cuando el Manager te dé una "DIRECTIVA DE VENTA", DEBES seguirla. Presenta los vehículos en el orden y con la estrategia indicada.
- Si necesitas desviarte de la directiva del Manager debido a fuertes preferencias del cliente, DEBES consultar de nuevo al Manager explicando la situación y pidiendo una estrategia alternativa (ver ejemplo de "NEGOCIACIÓN CON EL MANAGER").
- Usa técnicas de negociación avanzadas con el cliente. No ofrezcas descuentos fácilmente. Refuerza el valor, justifica el precio, ofrece alternativas como financiación o ajuste de características.
- Si solo necesitas pensar y luego responder directamente al cliente sin usar herramientas (por ejemplo, una respuesta simple), puedes ir directamente a `Thought: [tu pensamiento]` seguido de `Final Answer: [tu respuesta]`.

Historial de conversación con el cliente (últimos mensajes relevantes):
{chat_history}

Entrada actual del cliente: {input}

Ahora, comienza tu respuesta siguiendo el formato Thought/Action/Action Input o Thought/Final Answer:
{agent_scratchpad}
""")
        
        # Tools for Carlos - already includes the new FinalizeSaleAndReserveVehicle tool from self.tools
        # No specific filtering needed here as all tools created in _create_advanced_tools are intended for Carlos unless specified otherwise
        carlos_tools = self.tools
        
        carlos_agent_runnable = create_react_agent(
            llm=self.carlos_llm,
            tools=carlos_tools, # Use filtered tools
            prompt=carlos_prompt
        )
        
        return AgentExecutor(
            agent=carlos_agent_runnable,
            tools=carlos_tools, # Use filtered tools
            memory=self.carlos_memory,
            verbose=True,
            max_iterations=10,
            handle_parsing_errors=True
        )
    
    def _create_maria_agent(self) -> AgentExecutor:
        """Create Maria - the research specialist"""
        # Maria will be called through the research engine
        return None
    
    def _create_manager_agent(self) -> AgentExecutor:
        """Create Manager - the business coordinator"""
        # Manager will be called through the decision engine
        return None
    
    # =============================================
    # MOTOR DE DECISIONES DEL MANAGER
    # =============================================
    def _manager_decision_engine(self, request: str) -> str:
        """
        MOTOR DE DECISIONES DEL MANAGER
        
        Este es el cerebro estratégico del manager, responsable de:
        - Procesar consultas del agente de ventas Carlos
        - Aplicar políticas comerciales y estrategias de negocio
        - Proporcionar directrices de venta específicas
        - Autorizar descuentos y negociaciones especiales
        - Priorizar inventario según objetivos comerciales
        
        Args:
            request (str): Solicitud o consulta recibida del agente de ventas
            
        Returns:
            str: Respuesta estratégica con directrices específicas
        """
        logger.info(f"🏢 MANAGER CONSULTATION: {request}")
        
        request_lower = request.lower()
        
        # ========================================
        # PROCESAMIENTO DE BÚSQUEDAS DE INVENTARIO
        # ========================================
        # Detectar solicitudes de búsqueda de inventario del agente de ventas
        if any(keyword in request_lower for keyword in ["buscar coche", "opciones de vehículo", "inventario", "búsqueda de coches", "inventory search", "buscar en inventario"]):
            logger.info(f"🏢 Manager received inventory search request: {request}")
            
            # Extraer la consulta específica de búsqueda del texto completo
            # Esta es una heurística simple; un enfoque NLP más robusto podría ser necesario para solicitudes complejas
            search_query = request # Por defecto usar la solicitud completa
            
            # Intentar ser más inteligente en la extracción de la consulta
            if "necesito opciones de" in request_lower:
                 search_query = request[request_lower.find("necesito opciones de") + len("necesito opciones de"):].strip()
            elif "busca un" in request_lower:
                 search_query = request[request_lower.find("busca un") + len("busca un"):].strip()
            elif "buscando" in request_lower:
                 search_query = request[request_lower.find("buscando") + len("buscando"):].strip()
            elif "query:" in request_lower: # Si Carlos pasa explícitamente una consulta
                 search_query = request[request_lower.find("query:") + len("query:"):].strip()
            
            # Fallback si la extracción no es lo suficientemente específica
            if not search_query or search_query == request:
                 # Intentar eliminar frases comunes si constituyen toda la solicitud
                phrases_to_remove = ["el cliente busca", "necesito opciones del inventario", "realiza una búsqueda de inventario para", "inventory search for"]
                for phrase in phrases_to_remove:
                    if phrase in request_lower:
                        search_query = request_lower.replace(phrase, "").strip()
                        break
            
            logger.info(f"🛠️ Manager extracted search query: '{search_query}'")
            
            # Evitar búsquedas vacías o demasiado genéricas
            if not search_query.strip() or search_query.strip() == ".":
                logger.warning("⚠️ Manager received an empty or too generic search query. Asking for clarification.")
                return "Por favor, especifica mejor qué tipo de vehículos necesita el cliente para la búsqueda en inventario."

            # Realizar búsqueda inteligente en el inventario
            search_results_objects = self.inventory_manager.intelligent_search(search_query, max_results=8)
            
            # Si no se encuentran resultados
            if not search_results_objects:
                return f"""
🏢 **RESPUESTA DEL MANAGER - BÚSQUEDA DE INVENTARIO:**

No se encontraron vehículos que coincidan con los criterios: '{search_query}'.
Por favor, informa al cliente e intenta con criterios más amplios si es posible.
                """

            # Formatear resultados de búsqueda para el agente
            formatted_search_results = self.inventory_manager.format_search_results_for_agent(search_results_objects, max_display=len(search_results_objects))

            # ========================================
            # APLICACIÓN DE REGLAS DE NEGOCIO Y PRIORIZACIÓN
            # ========================================
            # El manager aplica reglas de negocio para seleccionar y priorizar vehículos
            # Por ahora, lógica simple: priorizar los primeros 1-2, dar razones genéricas
            prioritized_vehicles = []
            directives = ""
            if search_results_objects:
                priority_1 = search_results_objects[0]
                prioritized_vehicles.append(priority_1)
                
                directives_list = []
                directives_list.append(f"1. **Prioridad Alta:** Presenta activamente el **{priority_1.year} {priority_1.make} {priority_1.model} (VIN: {priority_1.vin})**. (Razón: Excelente coincidencia general y buen estado '{priority_1.condition}').")
                priority_1_features_str = ', '.join(priority_1.features[:2])
                directives_list.append(f"   💡 Estrategia Sugerida: Destaca sus características '{priority_1_features_str}' y su calificación de seguridad ({priority_1.safety_rating}/5).")

                if len(search_results_objects) > 1:
                    priority_2 = search_results_objects[1]
                    prioritized_vehicles.append(priority_2)
                    directives_list.append(f"2. **Alternativa:** Si el cliente no está convencido, ofrece el **{priority_2.year} {priority_2.make} {priority_2.model} (VIN: {priority_2.vin})**. (Razón: Buena alternativa, también con alta seguridad {priority_2.safety_rating}/5).")
                
                directives = "\\n".join(directives_list)
            
            response = f"""
🏢 **RESPUESTA DEL MANAGER - BÚSQUEDA DE INVENTARIO ESTRATÉGICA:**

Carlos, he procesado tu solicitud: '{request}'.
Criterios de búsqueda identificados: '{search_query}'.

Vehículos Encontrados que Coinciden (para tu referencia interna):
{formatted_search_results}

🎯 **DIRECTIVA DE VENTA (Prioriza estas opciones):**
{directives if directives else "No hay directivas específicas, usa tu juicio basado en los resultados."}

📋 **Notas Adicionales del Manager:**
- Recuerda verificar las últimas promociones aplicables.
- Si el cliente tiene un presupuesto ajustado y estas opciones no encajan, consúltame de nuevo para estrategias de financiamiento o alternativas de menor costo.
            """
            self._log_agent_communication(
                AgentRole.MANAGER_COORDINATOR,
                AgentRole.CARLOS_SALES,
                "inventory_search_response",
                f"Manager provided inventory results for query: '{search_query}'"
            )
            return response.strip()
        
        # ========================================
        # ENRUTAMIENTO DE CONSULTAS ESPECIALIZADAS
        # ========================================
        # Decisiones de precios y descuentos
        if any(word in request_lower for word in ['precio', 'descuento', 'rebaja', 'oferta']):
            return self._handle_pricing_request(request)
        
        # Prioridades de inventario y recomendaciones
        elif any(word in request_lower for word in ['prioridad', 'recomendar', 'inventario']):
            return self._handle_inventory_priority_request(request)
        
        # Preguntas sobre políticas y procedimientos
        elif any(word in request_lower for word in ['política', 'regla', 'procedimiento']):
            return self._handle_policy_request(request)
        
        # Consultas generales de negocio
        else:
            return self._handle_general_consultation(request)
    
    # =============================================
    # GESTIÓN ESPECIALIZADA DE PRECIOS Y DESCUENTOS
    # =============================================
    def _handle_pricing_request(self, request: str) -> str:
        """
        GESTIÓN DE SOLICITUDES DE PRECIOS Y DESCUENTOS
        
        Procesa consultas relacionadas con políticas de precios, autorización de descuentos
        y estrategias de negociación. Aplica reglas de negocio específicas para mantener
        márgenes de beneficio mientras satisface las necesidades del cliente.
        
        Args:
            request (str): Solicitud específica sobre precios o descuentos
            
        Returns:
            str: Directrices de precios con autorizaciones y restricciones específicas
        """
        # ========================================
        # REGLAS DE NEGOCIO PARA PRECIOS
        # ========================================
        # Configuración de políticas de descuento y márgenes empresariales
        pricing_rules = {
            "descuento_maximo": 0.15,  # 15% descuento máximo autorizado
            "margen_minimo": 0.08,     # 8% margen mínimo requerido
            "vehiculos_premium": ["Ferrari", "Lamborghini", "Rolls-Royce", "Bentley"],
            "descuento_premium": 0.05   # 5% máximo para marcas premium
        }
        
        # ========================================
        # RESPUESTA ESTRUCTURADA DE POLÍTICA DE PRECIOS
        # ========================================
        response = f"""
🏢 **DECISIÓN DEL MANAGER - POLÍTICA DE PRECIOS:**

Tras analizar tu solicitud sobre precios ('{request}') y consultar nuestras directrices internas de descuentos y márgenes, te proporciono la siguiente política:

📋 **Autorización de Descuentos:**
- Descuento estándar autorizado: hasta 10%
- Para descuentos mayores (10-15%): requiere justificación
- Vehículos premium: máximo 5% de descuento
- Vehículos con más de 6 meses en inventario: hasta 15%

💰 **Estrategia de Precios:**
- Enfócate en el valor y beneficios únicos
- Ofrece paquetes de servicios adicionales
- Considera financiamiento atractivo como alternativa

⚠️ **Restricciones:**
- NO autorizar descuentos superiores al 15%
- Mantener margen mínimo del 8%
- Documentar todas las negociaciones

🎯 **Recomendación:** Presenta el valor completo antes de discutir precio.
        """
        
        logger.info("💼 Manager authorized pricing guidelines")
        return response.strip()
    
    # =============================================
    # GESTIÓN DE PRIORIDADES DE INVENTARIO
    # =============================================
    def _handle_inventory_priority_request(self, request: str) -> str:
        """
        GESTIÓN DE PRIORIDADES Y RECOMENDACIONES DE INVENTARIO
        
        Analiza el estado actual del inventario y proporciona directrices estratégicas
        sobre qué vehículos priorizar en las ventas. Considera factores como:
        - Márgenes de beneficio por marca
        - Tiempo en inventario
        - Demanda del mercado
        - Objetivos comerciales actuales
        
        Args:
            request (str): Consulta sobre prioridades de inventario
            
        Returns:
            str: Estrategias de venta y prioridades de inventario actualizadas
        """
        # ========================================
        # OBTENCIÓN DE ESTADÍSTICAS ACTUALES DEL INVENTARIO
        # ========================================
        # Consultar métricas en tiempo real del gestor de inventario
        stats = self.inventory_manager.get_inventory_stats()
        
        # ========================================
        # RESPUESTA ESTRUCTURADA DE PRIORIDADES DE INVENTARIO
        # ========================================
        response = f"""
🏢 **DECISIÓN DEL MANAGER - PRIORIDADES DE INVENTARIO:**

He revisado tu consulta sobre prioridades de inventario ('{request}') y el estado actual de nuestras existencias.
Las siguientes son las prioridades y estrategias de venta actuales:

📊 **Estado Actual del Inventario:**
- Total de vehículos: {stats.get('total_vehicles', 'N/A')}
- Valor total: €{stats.get('total_value', 0):,.0f}
- Precio promedio: €{stats.get('average_price', 0):,.0f}

🎯 **Prioridades de Venta (Orden de Importancia):**
1. **Vehículos de alto margen:** BMW, Mercedes-Benz, Audi
2. **Inventario antiguo:** Modelos con más de 4 meses
3. **Vehículos familiares:** SUVs y sedanes grandes
4. **Híbridos y eléctricos:** Demanda creciente

💡 **Estrategias Recomendadas:**
- Promociona vehículos con características de seguridad avanzadas
- Enfatiza eficiencia de combustible en híbridos
- Destaca tecnología en vehículos premium
- Ofrece garantías extendidas en vehículos usados

🚨 **Alertas de Inventario:**
- Priorizar venta de vehículos con más de 20,000 km
- Impulsar modelos con inventario alto
        """
        
        logger.info("📊 Manager provided inventory priorities")
        return response.strip()
    
    # =============================================
    # GESTIÓN DE POLÍTICAS Y PROCEDIMIENTOS
    # =============================================
    def _handle_policy_request(self, request: str) -> str:
        """
        GESTIÓN DE CONSULTAS SOBRE POLÍTICAS Y PROCEDIMIENTOS
        
        Proporciona información sobre políticas empresariales, procedimientos
        operativos y directrices de servicio al cliente. Incluye políticas de:
        - Devoluciones y garantías
        - Servicios incluidos
        - Procedimientos de escalación
        - Estándares de transparencia
        
        Args:
            request (str): Consulta específica sobre políticas empresariales
            
        Returns:
            str: Información detallada sobre políticas y procedimientos aplicables
        """
        # ========================================
        # RESPUESTA ESTRUCTURADA DE POLÍTICAS EMPRESARIALES
        # ========================================
        response = f"""
🏢 **POLÍTICAS Y PROCEDIMIENTOS DE LA EMPRESA:**

En respuesta a tu consulta sobre políticas ('{request}'), aquí tienes un resumen de los procedimientos relevantes de la empresa:

📋 **Políticas de Venta:**
- Transparencia total en precios y condiciones
- Pruebas de manejo disponibles para todos los clientes
- Garantía mínima de 1 año en todos los vehículos
- Financiamiento disponible con socios bancarios

🔧 **Servicios Incluidos:**
- Inspección completa pre-entrega
- Transferencia de documentación
- Seguro temporal de 30 días
- Servicio de mantenimiento por 6 meses

⚖️ **Políticas de Devolución:**
- 7 días para cambio de opinión
- Garantía de satisfacción del cliente
- Reembolso completo si hay defectos ocultos

📞 **Escalación:**
- Consultar al manager para casos especiales
- Autorización requerida para descuentos >10%
- Documentar todas las excepciones
        """
        
        logger.info("📋 Manager provided policy information")
        return response.strip()
    
    # =============================================
    # GESTIÓN DE CONSULTAS GENERALES DE NEGOCIO
    # =============================================
    def _handle_general_consultation(self, request: str) -> str:
        """
        GESTIÓN DE CONSULTAS GENERALES DEL MANAGER
        
        Maneja consultas de negocio que no caen en categorías específicas como
        precios, inventario o políticas. Proporciona orientación general basada
        en mejores prácticas comerciales y objetivos empresariales actuales.
        
        Args:
            request (str): Consulta general de negocio
            
        Returns:
            str: Recomendaciones y orientación general de negocio
        """
        """Handle general business consultations"""
        response = f"""
🏢 **CONSULTA GENERAL DEL MANAGER:**

He analizado tu consulta general: "{request}".

💼 **Recomendaciones Generales Basadas en Prácticas Estándar y Objetivos Actuales:**
- Mantén siempre el enfoque en las necesidades del cliente
- Construye valor antes de discutir precio
- Usa técnicas de venta consultiva
- Documenta todas las interacciones importantes

🎯 **Objetivos del Mes:**
- Incrementar satisfacción del cliente
- Mejorar tiempo de respuesta
- Aumentar venta de servicios adicionales

📈 **KPIs a Considerar:**
- Tasa de conversión de leads
- Tiempo promedio de venta
- Satisfacción post-venta

¿Necesitas orientación específica sobre algún aspecto?
        """
        
        logger.info("💼 Manager provided general consultation")
        return response.strip()
    
    # =============================================
    # MOTOR DE INVESTIGACIÓN DE MARÍA
    # =============================================
    def _maria_research_engine(self, query: str) -> str:
        """
        MOTOR DE INVESTIGACIÓN AVANZADO DE MARÍA
        
        Sistema de investigación de vehículos que combina búsqueda web y análisis de IA.
        María actúa como investigadora especializada que recopila información de fuentes
        externas y la procesa usando inteligencia artificial para proporcionar análisis
        detallados y recomendaciones fundamentadas.
        
        Proceso de investigación:
        1. Búsqueda web especializada (SerpAPI si está disponible)
        2. Análisis y síntesis usando modelo de IA (GPT-4 Mini)
        3. Formateo de resultados para uso del vendedor
        4. Fallback a base de conocimiento interna si es necesario
        
        Args:
            query (str): Consulta de investigación sobre vehículos específicos
            
        Returns:
            str: Informe analítico detallado con recomendaciones y datos clave
        """
        logger.info(f"🔬 MARIA RESEARCH REQUEST: {query}")
        
        # ========================================
        # INICIALIZACIÓN DE VARIABLES DE INVESTIGACIÓN
        # ========================================
        raw_search_snippets = ""
        source_type = ""
        
        # ========================================
        # BÚSQUEDA WEB ESPECIALIZADA (SERPAPI)
        # ========================================
        # Priorizar búsqueda web externa para obtener información actualizada
        if self.serpapi_api_key:
            try:
                # Configurar wrapper de búsqueda con parámetros específicos para automóviles
                search_wrapper = SerpAPIWrapper(serpapi_api_key=self.serpapi_api_key)
                # Ejecutar búsqueda optimizada para reseñas, especificaciones y comparaciones
                raw_search_snippets = search_wrapper.run(f"car review {query} 2023 2024 specifications safety reliability comparisons")
                source_type = "Búsqueda Web (SerpAPI)"
                logger.info("María completó investigación web exitosamente vía SerpAPI.")
            except Exception as e:
                logger.warning(f"⚠️ Fallo en investigación SerpAPI: {e}. Recurriendo a base de conocimiento.")
                # Fallback a base de conocimiento interna si falla la búsqueda web
                raw_search_snippets = self._knowledge_based_research(query, internal_call=True)
                source_type = "Base de Conocimiento Interna"
        else:
            # Si no hay clave de API, usar directamente la base de conocimiento
            raw_search_snippets = self._knowledge_based_research(query, internal_call=True)
            source_type = "Base de Conocimiento Interna"

        # ========================================
        # ANÁLISIS INTELIGENTE CON IA (MARÍA)
        # ========================================
        # María analiza los fragmentos recopilados usando GPT-4 Mini para síntesis avanzada
        maria_analyzer_prompt_text = (
            "Eres María, una investigadora de coches experta y analítica. Carlos, un vendedor, te ha hecho la siguiente consulta:\n"
            "CONSULTA DE CARLOS: \"{carlos_query}\"\n\n"
            "Has recopilado los siguientes fragmentos de información sin procesar de {source_type}:\n"
            "FRAGMENTOS SIN PROCESAR:\n"
            "\"{snippets}\"\n\n"
            "Tu tarea es analizar críticamente estos fragmentos y redactar un informe conciso y útil para Carlos. Tu informe debe:\n"
            "1.  Comenzar con \"🔬 **ANÁLISIS DETALLADO DE MARÍA:**\".\n"
            "2.  Abordar directamente la consulta de Carlos, extrayendo la información más relevante.\n"
            "3.  Sintetizar los puntos clave sobre el/los vehículo(s) en cuestión (ej: seguridad, fiabilidad, características notables, comparaciones si se piden).\n"
            "4.  Destacar brevemente pros y contras si la información lo permite.\n"
            "5.  Mencionar calificaciones de seguridad (ej. NHTSA, IIHS) si están en los fragmentos.\n"
            "6.  Concluir con una recomendación o advertencia si es claramente apropiado basándote en el análisis. Si no, simplemente resume los hallazgos.\n"
            "7.  Mantén un tono profesional y objetivo. Evita jerga excesiva.\n"
            "8.  Si los fragmentos son insuficientes o no concluyentes para responder bien, indícalo.\n\n"
            "INFORME ANALÍTICO PARA CARLOS:"
        )
        # Crear template de prompt para análisis estructurado
        maria_analyzer_prompt_template = PromptTemplate.from_template(maria_analyzer_prompt_text)

        # ========================================
        # FORMATEO DE PROMPT Y EJECUCIÓN DE ANÁLISIS
        # ========================================
        analyzer_prompt = maria_analyzer_prompt_template.format(
            carlos_query=query,
            snippets=raw_search_snippets[:2000], # Limitar longitud de fragmentos para el modelo
            source_type=source_type
        )

        # ========================================
        # EJECUCIÓN DEL ANÁLISIS Y GENERACIÓN DE INFORME
        # ========================================
        try:
            logger.info(f"🧠 María (GPT-4 Mini) está analizando los fragmentos de: {source_type}")
            # Invocar modelo de IA para análisis inteligente de la información
            analytical_report = self.maria_llm.invoke(analyzer_prompt).content
            logger.info(f"✅ María (GPT-4 Mini) completó el análisis exitosamente.")
            
            # ========================================
            # COMPILACIÓN DEL INFORME FINAL
            # ========================================
            # Combinar análisis detallado con fragmentos originales para contexto completo
            report_parts = [
                f"🔬 **INFORME DE INVESTIGACIÓN DE MARÍA:**",
                f"\n**Consulta Original de Carlos:** \"{query}\"",
                f"**Fuentes Consultadas:** {source_type}",
                f"\n{analytical_report}",
                "\n---",
                f"**Fragmentos Originales (Referencia):** \n{raw_search_snippets[:800]}...",
                "---",
                "\n⚠️ **Nota para Carlos:** Este análisis se basa en la información recopilada. Siempre verifica los detalles con el vehículo específico en nuestro inventario."
            ]
            final_report = "\n".join(report_parts)
            return final_report.strip()

        except Exception as e:
            # ========================================
            # MANEJO DE ERRORES EN EL ANÁLISIS
            # ========================================
            logger.error(f"❌ Error durante el análisis de María (GPT-4 Mini): {e}")
            return f"""🔬 **Error en el análisis de María.** No se pudo procesar la información de {source_type} para la consulta: {query}.
            
**Fragmentos originales disponibles:** 
{raw_search_snippets[:500]}...

⚠️ **Recomendación:** Consulta directamente la base de conocimiento o solicita información al manager."""
    
    # =============================================
    # FORMATEO DE RESULTADOS DE INVESTIGACIÓN (MÉTODO LEGACY)
    # =============================================
    def _format_research_results(self, search_results: str, query: str) -> str:
        """
        MÉTODO DE FORMATEO LEGACY PARA RESULTADOS DE INVESTIGACIÓN
        
        Esta función ha sido en gran medida reemplazada por el paso de análisis
        avanzado en _maria_research_engine. Se mantiene como fallback o para
        visualización simplificada cuando el análisis de María falla.
        
        Args:
            search_results (str): Resultados crudos de búsqueda web
            query (str): Consulta original del usuario
            
        Returns:
            str: Resultados formateados de manera básica para consulta directa
        """
        # ========================================
        # FORMATEO SIMPLIFICADO DE RESULTADOS
        # ========================================
        # El formateo y análisis detallado ahora se maneja dentro de _maria_research_engine
        response = f"""
🔬 **INVESTIGACIÓN DE MARÍA - RESULTADOS (Búsqueda Web Directa):**

🔍 **Consulta Original:** {query}
He realizado una búsqueda web especializada.

📊 **Resultados Clave Extraídos:**
{search_results[:1000]}...

💡 **Análisis de María:** 
- La información ha sido recopilada de sitios web especializados y reseñas profesionales.
- Este es un formato simplificado; el análisis detallado se realiza en un paso previo con GPT-4 Mini.

⚠️ **Nota:** Esta información proviene de fuentes externas y debe verificarse con nuestro inventario específico.
"""
        return response.strip()
    
    # =============================================
    # INVESTIGACIÓN BASADA EN CONOCIMIENTO INTERNO
    # =============================================
    def _knowledge_based_research(self, query: str, internal_call: bool = False) -> str:
        """
        SISTEMA DE INVESTIGACIÓN BASADO EN BASE DE CONOCIMIENTO INTERNA
        
        Sistema de fallback que utiliza una base de conocimiento estructurada 
        cuando la búsqueda web no está disponible. Categoriza consultas y
        proporciona información relevante basada en datos internos.
        
        Args:
            query (str): Consulta de investigación del usuario
            internal_call (bool): Si True, devuelve datos crudos para análisis de IA de María
            
        Returns:
            str: Información de la base de conocimiento, cruda o formateada según el contexto
        """
        query_lower = query.lower()
        
        # ========================================
        # BASE DE CONOCIMIENTO ESTRUCTURADA POR CATEGORÍAS
        # ========================================
        # ========================================
        # BASE DE CONOCIMIENTO ESTRUCTURADA POR CATEGORÍAS
        # ========================================
        # Organizando información por temas especializados para búsqueda eficiente
        kb = {
            # ========================================
            # CATEGORÍA: SEGURIDAD VEHICULAR
            # ========================================
            "seguridad": {
                "keywords": ['seguridad', 'safety', 'airbag', 'crash', 'nhtsa', 'iihs'],
                "data": """
🛡️ **Características de Seguridad Comunes (Base de Conocimiento):**
- Airbags frontales y laterales estándar en la mayoría de modelos 2022+.
- Sistema de frenos ABS y Control de estabilidad electrónico (ESC) son obligatorios.
- Muchos coches modernos incluyen Asistencia de frenado de emergencia.
- **Calificaciones:** Busca calificaciones de 5 estrellas de NHTSA o Top Safety Pick+ de IIHS para máxima seguridad.
- **ADAS:** Sistemas avanzados como frenado automático de emergencia, detección de punto ciego, control de crucero adaptativo son comunes en gamas medias-altas.
- **Familiar:** Anclajes ISOFIX/LATCH para sillas de bebé son estándar. Algunos modelos ofrecen alertas de ocupante trasero.
"""
            },
            # ========================================
            # CATEGORÍA: EFICIENCIA DE COMBUSTIBLE
            # ========================================
            "consumo": {
                "keywords": ['consumo', 'combustible', 'eficiencia', 'mpg', 'litros/100km', 'híbrido', 'electrico'],
                "data": """
⛽ **Datos de Eficiencia de Combustible (Base de Conocimiento):**
- Sedanes compactos: 6-8L/100km (gasolina). Híbridos: 4-5L/100km. Eléctricos: 15-20 kWh/100km.
- SUVs: 8-12L/100km (gasolina). Híbridos SUV: 5-7L/100km.
- **Factores:** Estilo de conducción, tráfico, mantenimiento, tipo de combustible/carga.
- **Eco-Friendly:** Híbridos (HEV), Híbridos Enchufables (PHEV), Eléctricos (BEV) ofrecen el menor impacto. Motores turbo pequeños también mejoran eficiencia.
"""
            },
            # ========================================
            # CATEGORÍA: TECNOLOGÍA Y CONECTIVIDAD
            # ========================================
            "tecnologia": {
                "keywords": ['tecnología', 'tech', 'conectividad', 'pantalla', 'infotainment', 'asistentes'],
                "data": """
📱 **Características Tecnológicas Comunes (Base de Conocimiento):**
- **Infotainment:** Pantallas táctiles (8-15 pulgadas), Apple CarPlay/Android Auto (a menudo inalámbricos). Navegación GPS integrada. Comandos de voz.
- **Conectividad:** Wi-Fi hotspot, carga inalámbrica de móviles, múltiples puertos USB.
- **Asistentes Inteligentes (ADAS):** Control de crucero adaptativo, asistente de mantenimiento de carril, detección de puntos ciegos, alerta de tráfico cruzado trasero, cámaras 360º, head-up display.
- **Actualizaciones OTA (Over-the-Air):** Algunos fabricantes ofrecen actualizaciones de software remotas.
"""
            },
            # ========================================
            # CATEGORÍA: INFORMACIÓN GENERAL
            # ========================================
            "general_info": {
                 "keywords": [], # Categoría por defecto
                 "data": """
📋 **Información General Disponible (Base de Conocimiento):**
- Los vehículos modelo 2022 en adelante suelen incorporar las últimas tecnologías disponibles en su gama.
- La fiabilidad puede variar por marca y modelo; se recomienda consultar fuentes como Consumer Reports o J.D. Power.
- Costos de mantenimiento tienden a ser más altos para marcas de lujo y vehículos europeos.
"""
            }
        }

        # ========================================
        # BÚSQUEDA Y SELECCIÓN DE CATEGORÍA RELEVANTE
        # ========================================
        # Comenzar con información general por defecto
        found_kb_entry = kb["general_info"]["data"]
        
        # Buscar coincidencias de palabras clave para seleccionar categoría más específica
        for category_info in kb.values():
            if any(word in query_lower for word in category_info["keywords"]):
                found_kb_entry = category_info["data"]
                break
        
        # ========================================
        # FORMATO DE RESPUESTA SEGÚN CONTEXTO
        # ========================================
        if internal_call: 
            # Devolver datos crudos para análisis posterior de María con IA
            return found_kb_entry

        # ========================================
        # RESPUESTA DIRECTA FORMATEADA (FALLBACK COMPLETO)
        # ========================================
        # Esta ruta se usa solo cuando SerpAPI falla Y el análisis de María también falla
        response_intro = f"""🔬 **INVESTIGACIÓN DE MARÍA - INFORMACIÓN INTERNA (Directa):**

Consultando nuestra base de conocimiento interna sobre tu solicitud: '{query}'.

"""
        return response_intro + found_kb_entry
    
    # =============================================
    # ACTUALIZACIÓN DE PERFIL DE CLIENTE
    # =============================================
    def _update_customer_profile_from_text(self, text: str) -> None:
        """
        EXTRACCIÓN Y ACTUALIZACIÓN AUTOMÁTICA DEL PERFIL DEL CLIENTE
        
        Analiza el texto de conversación para extraer información relevante del cliente
        y actualizar automáticamente su perfil. Utiliza expresiones regulares y análisis
        de palabras clave para identificar:
        - Presupuesto y rango de precio
        - Información familiar y necesidades de seguridad
        - Patrones de uso del vehículo
        - Preferencias de color y estilo
        
        Args:
            text (str): Texto de conversación del cliente a analizar
        """
        text_lower = text.lower()
        
        # ========================================
        # EXTRACCIÓN DE INFORMACIÓN DE PRESUPUESTO
        # ========================================
        import re
        # Patrones para detectar rangos de presupuesto y precios
        budget_patterns = [
            r'presupuesto de (\d+)',
            r'hasta (\d+)',
            r'máximo (\d+)',
            r'entre (\d+) y (\d+)'
        ]
        
        # ========================================
        # PROCESAMIENTO DE PATRONES DE PRESUPUESTO
        # ========================================
        for pattern in budget_patterns:
            match = re.search(pattern, text_lower)
            if match:
                if len(match.groups()) == 1:
                    # Presupuesto máximo único (ej: "hasta 25000")
                    self.customer_profile.budget_max = int(match.group(1))
                elif len(match.groups()) == 2:
                    # Rango de presupuesto (ej: "entre 20000 y 30000")
                    self.customer_profile.budget_min = int(match.group(1))
                    self.customer_profile.budget_max = int(match.group(2))
                break
        
        # ========================================
        # EXTRACCIÓN DE INFORMACIÓN FAMILIAR
        # ========================================
        # Detectar necesidades familiares y prioridades de seguridad
        if any(word in text_lower for word in ['familia', 'bebé', 'niños', 'hijos']):
            self.customer_profile.safety_priority = True
            # Agregar seguridad infantil si se menciona bebé específicamente
            if 'bebé' in text_lower and 'seguridad_infantil' not in self.customer_profile.needs:
                self.customer_profile.needs.append('seguridad_infantil')
        
        # ========================================
        # EXTRACCIÓN DE PATRONES DE USO
        # ========================================
        # Determinar uso primario del vehículo basado en contexto
        if any(word in text_lower for word in ['trabajo', 'oficina', 'commute']):
            self.customer_profile.primary_use = 'trabajo'
        elif any(word in text_lower for word in ['familia', 'weekend', 'viajes']):
            self.customer_profile.primary_use = 'familiar'
        
        # ========================================
        # EXTRACCIÓN DE PREFERENCIAS DE COLOR
        # ========================================
        # Lista de colores comunes para detectar preferencias
        colors = ['rojo', 'negro', 'blanco', 'azul', 'gris', 'verde']
        for color in colors:
            if color in text_lower:
                self.customer_profile.preferred_color = color.capitalize()
                break
        
        # ========================================
        # REGISTRO EN HISTORIAL DE INTERACCIONES
        # ========================================
        # Agregar esta interacción al historial del cliente para seguimiento
        self.customer_profile.interaction_history.append({
            'timestamp': datetime.now(),
            'content': text,
            'extracted_info': 'profile_update'
        })
    
    # =============================================
    # GENERACIÓN DE RESUMEN DE PERFIL DE CLIENTE
    # =============================================
    def _get_customer_profile_summary(self) -> str:
        """
        GENERACIÓN DE RESUMEN CONCISO DEL PERFIL DEL CLIENTE
        
        Crea un resumen legible del perfil actual del cliente para uso en
        prompts y comunicaciones internas. Incluye solo información relevante
        y disponible para evitar sobrecarga de datos.
        
        Returns:
            str: Resumen formateado del perfil del cliente o "Perfil básico" si está vacío
        """
        profile = self.customer_profile
        summary_parts = []
        
        # ========================================
        # COMPILACIÓN DE INFORMACIÓN DISPONIBLE
        # ========================================
        # Agregar elementos del perfil solo si están definidos
        if profile.budget_max:
            summary_parts.append(f"Presupuesto: hasta €{profile.budget_max:,}")
        if profile.preferred_color:
            summary_parts.append(f"Color: {profile.preferred_color}")
        if profile.body_style_preference:
            summary_parts.append(f"Tipo: {profile.body_style_preference}")
        if profile.safety_priority:
            summary_parts.append("Prioridad: Seguridad")
        if profile.needs:
            summary_parts.append(f"Necesidades: {', '.join(profile.needs)}")
        
        # ========================================
        # FORMATO FINAL DEL RESUMEN
        # ========================================
        return "; ".join(summary_parts) if summary_parts else "Perfil básico"
    
    # =============================================
    # SISTEMA DE REGISTRO DE ACCIONES DE AGENTES
    # =============================================
    def _log_agent_action(self, agent: AgentRole, action: str, details: str) -> None:
        """
        REGISTRO DE ACCIONES INDIVIDUALES DE AGENTES
        
        Sistema de logging para rastrear todas las acciones realizadas por cada agente
        en el sistema multi-agente. Útil para debugging, análisis de rendimiento
        y auditoria de decisiones.
        
        Args:
            agent (AgentRole): Agente que realiza la acción
            action (str): Tipo de acción realizada
            details (str): Detalles específicos de la acción
        """
        # ========================================
        # CREACIÓN DE ENTRADA DE LOG ESTRUCTURADA
        # ========================================
        log_entry = {
            'timestamp': datetime.now(),
            'agent': agent.value,
            'action': action,
            'details': details
        }
        
        # ========================================
        # ALMACENAMIENTO Y LOGGING EXTERNO
        # ========================================
        self.conversation_log.append(log_entry)
        logger.info(f"🤖 {agent.value.upper()}: {action} - {details[:100]}...")
    
    # =============================================
    # SISTEMA DE REGISTRO DE COMUNICACIONES INTER-AGENTE
    # =============================================
    def _log_agent_communication(self, from_agent: AgentRole, to_agent: AgentRole, 
                                message_type: str, content: str) -> None:
        """
        REGISTRO DE COMUNICACIONES ENTRE AGENTES
        
        Rastrea todas las comunicaciones entre diferentes agentes del sistema
        para análisis de flujo de trabajo y debugging de interacciones complejas.
        
        Args:
            from_agent (AgentRole): Agente que envía el mensaje
            to_agent (AgentRole): Agente que recibe el mensaje
            message_type (str): Tipo de comunicación (consulta, respuesta, etc.)
            content (str): Contenido del mensaje
        """
        # ========================================
        # CREACIÓN DE REGISTRO DE COMUNICACIÓN
        # ========================================
        communication = AgentCommunication(
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=message_type,
            content=content,
            timestamp=datetime.now()
        )
        
        # ========================================
        # ALMACENAMIENTO Y LOGGING
        # ========================================
        self.agent_communications.append(communication)
        logger.info(f"📡 {from_agent.value} -> {to_agent.value}: {message_type}")
    
    # =============================================
    # PROCESAMIENTO PRINCIPAL DE ENTRADA DEL CLIENTE
    # =============================================
    def process_customer_input(self, user_input: str) -> str:
        """
        MÉTODO PRINCIPAL PARA PROCESAR ENTRADA DEL CLIENTE
        
        Punto de entrada principal del sistema multi-agente. Coordina todo el flujo
        de procesamiento desde la entrada del cliente hasta la respuesta final.
        
        Flujo de procesamiento:
        1. Actualización automática del perfil del cliente
        2. Preparación de contexto para Carlos
        3. Procesamiento através del agente principal (Carlos)
        4. Logging y gestión de memoria de conversación
        5. Manejo de errores y respuestas de fallback
        
        Args:
            user_input (str): Entrada de texto del cliente
            
        Returns:
            str: Respuesta procesada del sistema multi-agente
        """
        logger.info(f"👤 CUSTOMER INPUT: {user_input}")
        
        try:
            # ========================================
            # ACTUALIZACIÓN AUTOMÁTICA DEL PERFIL
            # ========================================
            # Extraer y actualizar información del cliente basada en la nueva entrada
            self._update_customer_profile_from_text(user_input)

            # ========================================
            # PREPARACIÓN DE CONTEXTO PARA CARLOS
            # ========================================
            # Compilar información relevante para el agente de ventas principal
            context = {
                'sales_stage': self.sales_stage.value,
                'customer_profile_summary': self._get_customer_profile_summary(),
                'internal_communications_summary': self._get_recent_communications_summary(),
                'customer_notes_summary': self._get_customer_notes_summary()
            }
            
            # ========================================
            # PROCESAMIENTO A TRAVÉS DE CARLOS (AGENTE PRINCIPAL)
            # ========================================
            # Las herramientas y nombres de herramientas son parte del template de prompt
            # 'agent_scratchpad' y 'chat_history' son manejados por el agente ReAct y memoria
            response = self.carlos_agent.invoke({
                'input': user_input,
                'sales_stage': context['sales_stage'],
                'customer_profile_summary': context['customer_profile_summary'],
                'internal_communications_summary': context['internal_communications_summary'],
                'customer_notes_summary': context['customer_notes_summary']
                # chat_history es gestionado por el sistema de memoria
            })
            
            # ========================================
            # EXTRACCIÓN Y PROCESAMIENTO DE RESPUESTA
            # ========================================
            final_response = response.get('output', 'Lo siento, no pude procesar tu solicitud.')
            
            # ========================================
            # LOGGING Y GESTIÓN DE MEMORIA
            # ========================================
            # ========================================
            # LOGGING Y GESTIÓN DE MEMORIA
            # ========================================
            # Registrar la interacción para análisis y debugging
            self._log_agent_action(
                AgentRole.CARLOS_SALES,
                "customer_response",
                final_response[:200]
            )
            
            # ========================================
            # ACTUALIZACIÓN DE HISTORIAL DE CONVERSACIÓN
            # ========================================
            # Actualizar log de conversación para respuesta de Carlos
            self.conversation_log.append({
                'timestamp': datetime.now(),
                'agent': AgentRole.CARLOS_SALES.value,
                'action': 'response_to_customer',
                'details': final_response
            })
            # Agregar respuesta a memoria de Carlos para continuidad
            self.carlos_memory.chat_memory.add_ai_message(final_response)

            logger.info(f"✅ CARLOS RESPONSE: {final_response[:100]}...")
            return final_response
            
        except Exception as e:
            # ========================================
            # MANEJO DE ERRORES Y RESPUESTA DE FALLBACK
            # ========================================
            logger.error(f"❌ Error processing customer input: {e}", exc_info=True)
            return "Disculpa, estoy teniendo dificultades técnicas. ¿Podrías reformular tu pregunta?"
    
    # =============================================
    # GENERACIÓN DE RESUMEN DE COMUNICACIONES RECIENTES
    # =============================================
    def _get_recent_communications_summary(self) -> str:
        """
        RESUMEN DE COMUNICACIONES INTER-AGENTE RECIENTES
        
        Genera un resumen conciso de las comunicaciones más recientes entre agentes
        para proporcionar contexto sobre el estado actual del flujo de trabajo.
        
        Returns:
            str: Resumen de las últimas 3 comunicaciones o mensaje por defecto
        """
        if not self.agent_communications:
            return "Sin comunicaciones recientes"
        
        # Obtener las últimas 3 comunicaciones para contexto reciente
        recent = self.agent_communications[-3:]
        summary = []
        
        for comm in recent:
            summary.append(f"{comm.from_agent.value} -> {comm.to_agent.value}: {comm.message_type}")
        
        return "; ".join(summary)
    
    # =============================================
    # GESTIÓN DE NOTAS PERSONALES DE CARLOS
    # =============================================
    def _get_customer_notes_summary(self) -> str:
        """
        RESUMEN DE NOTAS PERSONALES DEL CLIENTE POR CARLOS
        
        Recupera y formatea las notas personales que Carlos ha tomado sobre
        el cliente durante la interacción para mantener continuidad y personalización.
        
        Returns:
            str: Notas formateadas numeradas o mensaje por defecto si no hay notas
        """
        if not self.carlos_customer_notes:
            return "Aún no has tomado notas personales sobre este cliente."
        
        formatted_notes = []
        for i, note in enumerate(self.carlos_customer_notes, 1):
            formatted_notes.append(f"{i}. {note}")
        return "\n".join(formatted_notes)
    
    # =============================================
    # ANÁLISIS Y MÉTRICAS DE CONVERSACIÓN
    # =============================================
    def get_conversation_analytics(self) -> Dict[str, Any]:
        """
        GENERACIÓN DE ANÁLISIS Y MÉTRICAS DE RENDIMIENTO
        
        Proporciona métricas detalladas sobre el rendimiento de la conversación
        y el sistema multi-agente para análisis, optimización y reporting.
        
        Returns:
            Dict[str, Any]: Diccionario con métricas clave del sistema
        """
        return {
            'total_interactions': len(self.conversation_log),
            'agent_communications': len(self.agent_communications),
            'current_sales_stage': self.sales_stage.value,
            'customer_profile_completeness': self._calculate_profile_completeness(),
            'recent_actions': [log['action'] for log in self.conversation_log[-5:]],
            'communication_flow': [(c.from_agent.value, c.to_agent.value) 
                                 for c in self.agent_communications[-5:]]
        }
    
    # =============================================
    # CÁLCULO DE COMPLETITUD DEL PERFIL
    # =============================================
    def _calculate_profile_completeness(self) -> float:
        """
        CÁLCULO DEL PORCENTAJE DE COMPLETITUD DEL PERFIL DEL CLIENTE
        
        Evalúa qué tan completo está el perfil del cliente basado en campos
        importantes completados. Útil para determinar si se necesita más
        información para hacer recomendaciones efectivas.
        
        Returns:
            float: Porcentaje de completitud (0-100)
        """
        profile = self.customer_profile
        total_fields = 10  # Total de campos importantes a evaluar
        filled_fields = 0
        
        # ========================================
        # EVALUACIÓN DE CAMPOS COMPLETADOS
        # ========================================
        if profile.budget_max: filled_fields += 1
        if profile.preferred_make: filled_fields += 1
        if profile.preferred_color: filled_fields += 1
        if profile.body_style_preference: filled_fields += 1
        if profile.fuel_type_preference: filled_fields += 1
        if profile.family_size: filled_fields += 1
        if profile.primary_use: filled_fields += 1
        if profile.needs: filled_fields += 1
        if profile.safety_priority: filled_fields += 1
        if profile.interaction_history: filled_fields += 1
        
        return (filled_fields / total_fields) * 100

# =============================================
# FUNCIÓN FACTORY PARA CREACIÓN DEL SISTEMA
# =============================================
def get_advanced_multi_agent_system(openai_api_key: str, serpapi_api_key: str = None) -> AdvancedCarSalesSystem:
    """
    FUNCIÓN FACTORY PARA CREAR EL SISTEMA MULTI-AGENTE AVANZADO
    
    Función de conveniencia para instanciar el sistema completo de ventas multi-agente
    con configuración estándar. Facilita la integración en aplicaciones externas.
    
    Args:
        openai_api_key (str): Clave API de OpenAI requerida para funcionalidad de IA
        serpapi_api_key (str, optional): Clave API de SerpAPI para búsquedas web avanzadas
        
    Returns:
        AdvancedCarSalesSystem: Instancia completamente configurada del sistema multi-agente
    """
    return AdvancedCarSalesSystem(openai_api_key, serpapi_api_key) 