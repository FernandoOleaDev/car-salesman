import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory

from enhanced_inventory_manager import get_inventory_manager, CarSearchResult
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('carbot_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SalesStage(Enum):
    GREETING = "greeting"
    DISCOVERY = "discovery"
    PRESENTATION = "presentation"
    OBJECTION_HANDLING = "objection_handling"
    NEGOTIATION = "negotiation"
    CLOSING = "closing"
    FOLLOW_UP = "follow_up"

class AgentRole(Enum):
    CARLOS_SALES = "carlos_sales"
    MARIA_RESEARCH = "maria_research"
    MANAGER_COORDINATOR = "manager_coordinator"

@dataclass
class CustomerProfile:
    """Comprehensive customer profile"""
    name: Optional[str] = None
    budget_min: Optional[int] = None
    budget_max: Optional[int] = None
    preferred_make: Optional[str] = None
    preferred_color: Optional[str] = None
    body_style_preference: Optional[str] = None
    fuel_type_preference: Optional[str] = None
    family_size: Optional[int] = None
    primary_use: Optional[str] = None
    safety_priority: bool = False
    luxury_preference: bool = False
    eco_friendly: bool = False
    needs: List[str] = None
    objections: List[str] = None
    interaction_history: List[Dict] = None
    
    def __post_init__(self):
        if self.needs is None:
            self.needs = []
        if self.objections is None:
            self.objections = []
        if self.interaction_history is None:
            self.interaction_history = []

@dataclass
class AgentCommunication:
    """Inter-agent communication structure"""
    from_agent: AgentRole
    to_agent: AgentRole
    message_type: str
    content: str
    timestamp: datetime
    priority: str = "normal"
    requires_response: bool = False

class AdvancedCarSalesSystem:
    """Advanced multi-agent car sales system with professional workflows"""
    
    def __init__(self, openai_api_key: str, serpapi_api_key: str = None):
        self.openai_api_key = openai_api_key
        self.serpapi_api_key = serpapi_api_key
        
        # Initialize inventory manager
        self.inventory_manager = get_inventory_manager()
        
        # Initialize customer profile
        self.customer_profile = CustomerProfile()
        self.sales_stage = SalesStage.GREETING
        
        # Communication system
        self.agent_communications = []
        self.conversation_log = []
        self.carlos_customer_notes: List[str] = [] # Carlos's customer notes
        
        # Initialize LLMs with latest models
        self.carlos_llm = ChatOpenAI(
            temperature=0.8,  # Creative for sales conversations
            openai_api_key=openai_api_key,
            model_name="gpt-4o",  # Latest GPT-4o for advanced sales
            max_tokens=1000
        )
        
        self.maria_llm = ChatOpenAI(
            temperature=1,  # Factual for research
            openai_api_key=openai_api_key,
            model_name="o4-mini",  # o4-mini for analytical research
            max_tokens=800
        )
        
        self.manager_llm = ChatOpenAI(
            temperature=0.4,  # Balanced for coordination
            openai_api_key=openai_api_key,
            model_name="gpt-4o",  # GPT-4o for intelligent coordination
            max_tokens=600
        )
        
        # Initialize memory for each agent
        self.carlos_memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            input_key="input",
            k=10,
            return_messages=True
        )
        
        # Initialize tools and agents
        self.tools = self._create_advanced_tools()
        self.carlos_agent = self._create_carlos_agent()
        self.maria_agent = self._create_maria_agent()
        self.manager_agent = self._create_manager_agent()
        
        logger.info("🚀 Advanced Car Sales System initialized successfully")
    
    def _perform_intelligent_inventory_search(self, query: str) -> str:
        """Helper method for intelligent inventory search, used by the Manager."""
        # No direct logging of agent action here, as it's an internal system capability
        # The manager's action will be logged when it decides to use this.
        try:
            results_objects = self.inventory_manager.intelligent_search(query, max_results=8)
            formatted_results = self.inventory_manager.format_search_results_for_agent(results_objects, max_display=len(results_objects))
            
            # Log the outcome of the search itself
            logger.info(f"⚙️ System performed inventory search for query '{query}', found {len(results_objects)} vehicles.")
            return formatted_results
        except Exception as e:
            logger.error(f"❌ Error in _perform_intelligent_inventory_search: {e}")
            return "❌ Error interno al realizar la búsqueda de inventario."
    
    def _create_advanced_tools(self) -> List[Tool]:
        """Create advanced tools for the multi-agent system"""
        tools = []
        
        # Manager consultation tool
        def consult_manager(request: str) -> str:
            """Consult with the manager for pricing, priorities, and business decisions"""
            self._log_agent_communication(
                AgentRole.CARLOS_SALES,
                AgentRole.MANAGER_COORDINATOR,
                "consultation_request",
                request
            )
            
            try:
                # Manager's business logic and decision making
                manager_response = self._manager_decision_engine(request)
                
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
        
        # Research tool via Maria
        def research_vehicle_info(query: str) -> str:
            """Research detailed vehicle information, reviews, and market data"""
            self._log_agent_communication(
                AgentRole.CARLOS_SALES,
                AgentRole.MARIA_RESEARCH,
                "research_request",
                query
            )
            
            try:
                research_result = self._maria_research_engine(query)
                
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
        
        # Customer profiling tool
        def update_customer_profile(info: str) -> str:
            """Update customer profile with new information"""
            self._log_agent_action(AgentRole.CARLOS_SALES, "profile_update", info)
            
            try:
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
        
        # Sales stage management
        def update_sales_stage(stage: str) -> str:
            """Update the current sales stage"""
            try:
                new_stage = SalesStage(stage.lower())
                old_stage = self.sales_stage
                self.sales_stage = new_stage
                
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

        # Tool to finalize sale and reserve vehicle
        def finalize_sale_and_reserve_vehicle(vin: str) -> str:
            """Finalizes the sale of a vehicle and marks it as reserved in the inventory. 
            Use this tool ONLY when the customer has explicitly confirmed they want to purchase a specific vehicle. 
            Provide the VIN of the vehicle to be reserved.

            Args:
                vin (str): The Vehicle Identification Number (VIN) of the car to be reserved.
            Returns:
                str: Confirmation message or error if reservation failed.
            """
            self._log_agent_action(AgentRole.CARLOS_SALES, "finalize_sale_attempt", f"VIN: {vin}")
            try:
                success = self.inventory_manager.reserve_vehicle(vin)
                if success:
                    # Potentially trigger other post-sale actions here in a real system
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

        # Tool to give final response to client
        def respond_to_client(response: str) -> str:
            """Delivers your message directly to the customer. Use this when you are ready to communicate your thought or answer.
            IMPORTANT: After using this tool, you MUST then use the 'Final Answer:' format to conclude your turn, typically repeating the response you just sent.

            Args:
                response (str): The complete message you want to send to the client.
            Returns:
                str: The response that was sent to the client. This observation will be your own message.
            """
            logger.info(f"🗣️ CARLOS TO CLIENT (via RespondToClient tool): {response[:100]}...")
            return response

        tools.append(Tool(
            name="RespondToClient",
            func=respond_to_client,
            description="Use this tool to provide your final answer or response directly to the customer. This action concludes your processing for the current customer input, and the observation returned will be the final answer."
        ))

        # Carlos's Customer Notes tool
        def update_customer_notes(note_to_add: str, mode: str = "append") -> str:
            """Adds or overwrites notes in Carlos's personal customer notepad. 
            Use 'append' to add a new note to existing ones. 
            Use 'overwrite' to replace all previous notes with the new note_to_add.
            This notepad is for capturing details, nuances, or specific customer statements that might not fit the structured profile.

            Args:
                note_to_add (str): The text of the note to add or use for overwriting.
                mode (str): Either 'append' or 'overwrite'. Defaults to 'append'.
            Returns:
                str: Confirmation message of the action taken.
            """
            self._log_agent_action(AgentRole.CARLOS_SALES, "update_customer_notes_attempt", f"Mode: {mode}, Note: {note_to_add[:50]}...")
            if mode.lower() == "overwrite":
                self.carlos_customer_notes = [note_to_add]
                logger.info(f"📝 Carlos's customer notes OVERWRITTEN. Current notes: {len(self.carlos_customer_notes)}")
                return f"Notas sobrescritas. Nueva nota: '{note_to_add[:100]}...'"
            elif mode.lower() == "append":
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
    
    def _manager_decision_engine(self, request: str) -> str:
        """Manager's decision-making engine for business policies"""
        logger.info(f"🏢 MANAGER CONSULTATION: {request}")
        
        request_lower = request.lower()
        
        # Inventory search request from Carlos
        if any(keyword in request_lower for keyword in ["buscar coche", "opciones de vehículo", "inventario", "búsqueda de coches", "inventory search", "buscar en inventario"]):
            logger.info(f"🏢 Manager received inventory search request: {request}")
            # Extract the actual query part for the inventory search.
            # This is a simple heuristic; a more robust NLP approach might be needed for complex requests.
            search_query = request # Default to full request
            # Try to be a bit smarter in extracting the query
            if "necesito opciones de" in request_lower:
                 search_query = request[request_lower.find("necesito opciones de") + len("necesito opciones de"):].strip()
            elif "busca un" in request_lower:
                 search_query = request[request_lower.find("busca un") + len("busca un"):].strip()
            elif "buscando" in request_lower:
                 search_query = request[request_lower.find("buscando") + len("buscando"):].strip()
            elif "query:" in request_lower: # If Carlos explicitly passes a query
                 search_query = request[request_lower.find("query:") + len("query:"):].strip()
            
            if not search_query or search_query == request: # Fallback if extraction is not specific enough
                 # Try to remove common phrases if they are the whole request
                phrases_to_remove = ["el cliente busca", "necesito opciones del inventario", "realiza una búsqueda de inventario para", "inventory search for"]
                for phrase in phrases_to_remove:
                    if phrase in request_lower:
                        search_query = request_lower.replace(phrase, "").strip()
                        break
            
            logger.info(f"🛠️ Manager extracted search query: '{search_query}'")
            if not search_query.strip() or search_query.strip() == ".": # Avoid empty searches
                logger.warning("⚠️ Manager received an empty or too generic search query. Asking for clarification.")
                return "Por favor, especifica mejor qué tipo de vehículos necesita el cliente para la búsqueda en inventario."

            search_results_objects = self.inventory_manager.intelligent_search(search_query, max_results=8) # Assuming this returns list of CarSearchResult
            
            if not search_results_objects:
                return f"""
🏢 **RESPUESTA DEL MANAGER - BÚSQUEDA DE INVENTARIO:**

No se encontraron vehículos que coincidan con los criterios: '{search_query}'.
Por favor, informa al cliente e intenta con criterios más amplios si es posible.
                """

            formatted_search_results = self.inventory_manager.format_search_results_for_agent(search_results_objects, max_display=len(search_results_objects))

            # Manager applies business rules to select and prioritize
            # For now, simple logic: prioritize first 1-2, give some generic reasons
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
        
        # Pricing decisions
        if any(word in request_lower for word in ['precio', 'descuento', 'rebaja', 'oferta']):
            return self._handle_pricing_request(request)
        
        # Inventory priorities
        elif any(word in request_lower for word in ['prioridad', 'recomendar', 'inventario']):
            return self._handle_inventory_priority_request(request)
        
        # Policy questions
        elif any(word in request_lower for word in ['política', 'regla', 'procedimiento']):
            return self._handle_policy_request(request)
        
        # General business consultation
        else:
            return self._handle_general_consultation(request)
    
    def _handle_pricing_request(self, request: str) -> str:
        """Handle pricing and discount requests"""
        # Business rules for pricing
        pricing_rules = {
            "descuento_maximo": 0.15,  # 15% max discount
            "margen_minimo": 0.08,     # 8% minimum margin
            "vehiculos_premium": ["Ferrari", "Lamborghini", "Rolls-Royce", "Bentley"],
            "descuento_premium": 0.05   # 5% max for premium brands
        }
        
        response = """
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
    
    def _handle_inventory_priority_request(self, request: str) -> str:
        """Handle inventory priority and recommendation requests"""
        # Get current inventory stats
        stats = self.inventory_manager.get_inventory_stats()
        
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
    
    def _handle_policy_request(self, request: str) -> str:
        """Handle policy and procedure questions"""
        response = """
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
    
    def _handle_general_consultation(self, request: str) -> str:
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
    
    def _maria_research_engine(self, query: str) -> str:
        """Maria's research engine for vehicle information"""
        logger.info(f"🔬 MARIA RESEARCH REQUEST: {query}")
        
        raw_search_snippets = ""
        source_type = ""
        
        # Try SerpAPI first if available
        if self.serpapi_api_key:
            try:
                search_wrapper = SerpAPIWrapper(serpapi_api_key=self.serpapi_api_key)
                raw_search_snippets = search_wrapper.run(f"car review {query} 2023 2024 specifications safety reliability comparisons")
                source_type = "Búsqueda Web (SerpAPI)"
                logger.info("Maria completed web research successfully via SerpAPI.")
            except Exception as e:
                logger.warning(f"⚠️ SerpAPI research failed: {e}. Falling back to knowledge base.")
                raw_search_snippets = self._knowledge_based_research(query, internal_call=True) # Get raw data
            source_type = "Base de Conocimiento Interna"

        # Now, Maria analyzes these snippets
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
        maria_analyzer_prompt_template = PromptTemplate.from_template(maria_analyzer_prompt_text)

        analyzer_prompt = maria_analyzer_prompt_template.format(
            carlos_query=query,
            snippets=raw_search_snippets[:2000], # Limit snippet length for the analyzer LLM
            source_type=source_type
        )

        try:
            logger.info(f"🧠 Maria (o4-mini) está analizando los fragmentos de: {source_type}")
            analytical_report = self.maria_llm.invoke(analyzer_prompt).content
            logger.info(f"✅ Maria (o4-mini) completó el análisis.")
            
            # Combine with original snippets for full context if needed, or just return report
            # For now, returning the detailed report Maria generated, plus context about sources.
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
            logger.error(f"❌ Error durante el análisis de María (o4-mini): {e}")
            return f"""🔬 Error en el análisis de María. No se pudo procesar la información de {source_type} para la consulta: {query}.
Fragmentos originales: {raw_search_snippets[:500]}...
"""
    
    def _format_research_results(self, search_results: str, query: str) -> str:
        # This function is largely superseded by the new analytical step in _maria_research_engine
        # However, it can be kept as a fallback or for simpler display if Maria's analysis fails.
        # For now, the main formatting is handled within _maria_research_engine itself.
        
        # Simplified version, as detailed formatting/analysis is now in _maria_research_engine
        response = f"""
🔬 **INVESTIGACIÓN DE MARÍA - RESULTADOS (Búsqueda Web Directa):**

🔍 **Consulta Original:** {query}
He realizado una búsqueda web.

📊 **Resultados Clave Extraídos:**
{search_results[:1000]}...

💡 **Análisis de María:** (Análisis más detallado ahora se realiza en un paso previo con o4-mini)
- La información ha sido recopilada de sitios web especializados y reseñas profesionales.

⚠️ **Nota:** Esta información proviene de fuentes externas.
"""
        return response.strip()
    
    def _knowledge_based_research(self, query: str, internal_call: bool = False) -> str:
        """Fallback knowledge-based research. If internal_call is True, returns raw-ish data for Maria's LLM to analyze."""
        query_lower = query.lower()
        
        # Knowledge base structured as a dictionary
        kb = {
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
            "general_info": {
                 "keywords": [], # Default
                 "data": """
📋 **Información General Disponible (Base de Conocimiento):**
- Los vehículos modelo 2022 en adelante suelen incorporar las últimas tecnologías disponibles en su gama.
- La fiabilidad puede variar por marca y modelo; se recomienda consultar fuentes como Consumer Reports o J.D. Power.
- Costos de mantenimiento tienden a ser más altos para marcas de lujo y vehículos europeos.
"""
            }
        }

        found_kb_entry = kb["general_info"]["data"] # Default
        for category_info in kb.values():
            if any(word in query_lower for word in category_info["keywords"]):
                found_kb_entry = category_info["data"]
                break
        
        if internal_call: # Return the raw-ish data for Maria's LLM
            return found_kb_entry

        # If not an internal call, means it's a direct fallback when SerpAPI is missing AND Maria's LLM analysis step is also skipped/failed
        # This path should ideally not be hit if Maria's LLM analysis works.
        response_intro = f"""🔬 **INVESTIGACIÓN DE MARÍA - INFORMACIÓN INTERNA (Directa):**\\n\\nConsultando nuestra base de conocimiento interna sobre tu solicitud: '{query}'.\\n"""
        return response_intro + found_kb_entry
    
    def _update_customer_profile_from_text(self, text: str) -> None:
        """Update customer profile from conversation text"""
        text_lower = text.lower()
        
        # Extract budget information
        import re
        budget_patterns = [
            r'presupuesto de (\d+)',
            r'hasta (\d+)',
            r'máximo (\d+)',
            r'entre (\d+) y (\d+)'
        ]
        
        for pattern in budget_patterns:
            match = re.search(pattern, text_lower)
            if match:
                if len(match.groups()) == 1:
                    self.customer_profile.budget_max = int(match.group(1))
                elif len(match.groups()) == 2:
                    self.customer_profile.budget_min = int(match.group(1))
                    self.customer_profile.budget_max = int(match.group(2))
                break
        
        # Extract family information
        if any(word in text_lower for word in ['familia', 'bebé', 'niños', 'hijos']):
            self.customer_profile.safety_priority = True
            if 'bebé' in text_lower and 'seguridad_infantil' not in self.customer_profile.needs:
                self.customer_profile.needs.append('seguridad_infantil')
        
        # Extract usage patterns
        if any(word in text_lower for word in ['trabajo', 'oficina', 'commute']):
            self.customer_profile.primary_use = 'trabajo'
        elif any(word in text_lower for word in ['familia', 'weekend', 'viajes']):
            self.customer_profile.primary_use = 'familiar'
        
        # Extract preferences
        colors = ['rojo', 'negro', 'blanco', 'azul', 'gris', 'verde']
        for color in colors:
            if color in text_lower:
                self.customer_profile.preferred_color = color.capitalize()
                break
        
        # Add to interaction history
        self.customer_profile.interaction_history.append({
            'timestamp': datetime.now(),
            'content': text,
            'extracted_info': 'profile_update'
        })
    
    def _get_customer_profile_summary(self) -> str:
        """Get a summary of the customer profile"""
        profile = self.customer_profile
        summary_parts = []
        
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
        
        return "; ".join(summary_parts) if summary_parts else "Perfil básico"
    
    def _log_agent_action(self, agent: AgentRole, action: str, details: str) -> None:
        """Log agent actions for debugging and analysis"""
        log_entry = {
            'timestamp': datetime.now(),
            'agent': agent.value,
            'action': action,
            'details': details
        }
        
        self.conversation_log.append(log_entry)
        logger.info(f"🤖 {agent.value.upper()}: {action} - {details[:100]}...")
    
    def _log_agent_communication(self, from_agent: AgentRole, to_agent: AgentRole, 
                                message_type: str, content: str) -> None:
        """Log inter-agent communications"""
        communication = AgentCommunication(
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=message_type,
            content=content,
            timestamp=datetime.now()
        )
        
        self.agent_communications.append(communication)
        logger.info(f"📡 {from_agent.value} -> {to_agent.value}: {message_type}")
    
    def process_customer_input(self, user_input: str) -> str:
        """Main method to process customer input through the multi-agent system"""
        logger.info(f"👤 CUSTOMER INPUT: {user_input}")
        
        try:
            # Update customer profile based on the new input
            self._update_customer_profile_from_text(user_input)

            # Prepare context for Carlos
            context = {
                'sales_stage': self.sales_stage.value,
                'customer_profile_summary': self._get_customer_profile_summary(),
                'internal_communications_summary': self._get_recent_communications_summary(),
                'customer_notes_summary': self._get_customer_notes_summary() # Add notes to context
            }
            
            # Process through Carlos (main sales agent)
            # The 'tools' and 'tool_names' are part of the prompt template itself.
            # 'agent_scratchpad' and 'chat_history' are handled by the ReAct agent and memory.
            response = self.carlos_agent.invoke({
                'input': user_input,
                'sales_stage': context['sales_stage'],
                'customer_profile_summary': context['customer_profile_summary'],
                'internal_communications_summary': context['internal_communications_summary'],
                'customer_notes_summary': context['customer_notes_summary']
                # chat_history is managed by memory
            })
            
            # Extract the final response
            final_response = response.get('output', 'Lo siento, no pude procesar tu solicitud.')
            
            # Log the interaction
            self._log_agent_action(
                AgentRole.CARLOS_SALES,
                "customer_response",
                final_response[:200]
            )
            
            # Update conversation log for Carlos's response
            self.conversation_log.append({
                'timestamp': datetime.now(),
                'agent': AgentRole.CARLOS_SALES.value,
                'action': 'response_to_customer',
                'details': final_response
            })
            self.carlos_memory.chat_memory.add_ai_message(final_response)


            logger.info(f"✅ CARLOS RESPONSE: {final_response[:100]}...")
            return final_response
            
        except Exception as e:
            logger.error(f"❌ Error processing customer input: {e}", exc_info=True)
            return "Disculpa, estoy teniendo dificultades técnicas. ¿Podrías reformular tu pregunta?"
    
    def _get_recent_communications_summary(self) -> str:
        """Get summary of recent inter-agent communications"""
        if not self.agent_communications:
            return "Sin comunicaciones recientes"
        
        recent = self.agent_communications[-3:]  # Last 3 communications
        summary = []
        
        for comm in recent:
            summary.append(f"{comm.from_agent.value} -> {comm.to_agent.value}: {comm.message_type}")
        
        return "; ".join(summary)
    
    def _get_customer_notes_summary(self) -> str:
        """Get a summary of Carlos's personal customer notes."""
        if not self.carlos_customer_notes:
            return "Aún no has tomado notas personales sobre este cliente."
        
        formatted_notes = []
        for i, note in enumerate(self.carlos_customer_notes, 1):
            formatted_notes.append(f"{i}. {note}")
        return "\n".join(formatted_notes)
    
    def get_conversation_analytics(self) -> Dict[str, Any]:
        """Get analytics about the conversation and agent performance"""
        return {
            'total_interactions': len(self.conversation_log),
            'agent_communications': len(self.agent_communications),
            'current_sales_stage': self.sales_stage.value,
            'customer_profile_completeness': self._calculate_profile_completeness(),
            'recent_actions': [log['action'] for log in self.conversation_log[-5:]],
            'communication_flow': [(c.from_agent.value, c.to_agent.value) 
                                 for c in self.agent_communications[-5:]]
        }
    
    def _calculate_profile_completeness(self) -> float:
        """Calculate how complete the customer profile is"""
        profile = self.customer_profile
        total_fields = 10  # Total important fields
        filled_fields = 0
        
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

def get_advanced_multi_agent_system(openai_api_key: str, serpapi_api_key: str = None) -> AdvancedCarSalesSystem:
    """Factory function to create the advanced multi-agent system"""
    return AdvancedCarSalesSystem(openai_api_key, serpapi_api_key) 