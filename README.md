# 🚗 CarBot Pro - Advanced Multi-Agent Car Sales System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-🦜🔗-green.svg)](https://langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-orange.svg)](https://openai.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-🎈-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A sophisticated multi-agent AI system for intelligent car sales, demonstrating advanced agent coordination, real-time research, and professional sales workflows.**

*Originally developed by Eduardo Hilario for AI Agents Day demo.*

---

## 🎯 Overview

CarBot Pro showcases the cutting-edge capabilities of multi-agent AI systems in a real-world sales scenario. The system coordinates three specialized AI agents to provide a complete, professional car buying experience with intelligent inventory management, real-time research, and advanced negotiation capabilities.

### 🤖 Meet the Agents

| Agent | Model | Role | Specialization |
|-------|-------|------|----------------|
| **🎯 Carlos** | GPT-4o | Sales Expert | Customer interaction, sales process, negotiation |
| **🔬 María** | o4-mini | Research Specialist | Vehicle research, technical analysis, market data |
| **🏢 Manager** | GPT-4o | Business Coordinator | Inventory management, pricing, business policies |

## ✨ Key Features

### 🔧 Advanced Capabilities
- **🧠 Intelligent Agent Coordination** - Seamless communication between specialized agents
- **🔍 Smart Inventory Search** - AI-powered vehicle matching with 40+ enriched vehicle database
- **🌐 Real-time Web Research** - Live vehicle information via SerpAPI integration
- **📊 Dynamic Customer Profiling** - Automatic extraction and management of customer preferences
- **💼 Professional Sales Process** - Complete sales funnel from greeting to closing
- **📈 Real-time Analytics** - Comprehensive metrics and conversation analytics
- **🔄 State Management** - Advanced sales stage tracking and progression

### 📋 Sales Process Stages
1. **Greeting** - Initial rapport building
2. **Discovery** - Needs assessment and profiling
3. **Presentation** - Intelligent vehicle recommendations
4. **Objection Handling** - Professional concern resolution
5. **Negotiation** - Policy-based pricing and alternatives
6. **Closing** - Sale finalization with inventory updates
7. **Follow-up** - Post-sale relationship management

### 🚗 Rich Vehicle Database
- **Comprehensive Data**: Make, model, year, color, mileage, price
- **Technical Specs**: Engine, transmission, fuel efficiency, safety ratings
- **Special Features**: Technology packages, interior details, location
- **Price Range**: €25,000 - €320,000 (economy to luxury supercars)

---

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
git clone https://github.com/LIDR-academy/car-salesman.git
cd car-salesman
python quick_setup_advanced.py
```

### Option 2: Manual Installation

#### Prerequisites
- Python 3.8 or higher
- OpenAI API key (required)
- SerpAPI key (optional, for web research)

#### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/LIDR-academy/car-salesman.git
cd car-salesman
```

2. **Create virtual environment**
```bash
python -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
cp config.env .env
# Edit .env with your API keys
```

5. **Set up API keys in `.env`**
```env
# Required: OpenAI API Key
OPENAI_API_KEY=sk-your_openai_key_here

# Optional: SerpAPI Key for web research
SERPAPI_API_KEY=your_serpapi_key_here
```

6. **Run the application**
```bash
streamlit run enhanced_app.py
```

7. **Access the interface**
   - Open your browser to `http://localhost:8501`
   - Configure API keys in the sidebar
   - Click "🚀 Initialize Advanced System"
   - Start chatting with Carlos!

---

## 📖 Usage Examples

### Basic Interaction Flow

```python
# Example conversation flow
customer_input = "I'm looking for a safe family car"

# System processes through multi-agent workflow:
# 1. Carlos updates customer profile
# 2. Carlos consults Manager for inventory options
# 3. Manager searches inventory and provides recommendations
# 4. Carlos presents options to customer
# 5. Customer asks for technical details
# 6. Carlos requests research from María
# 7. María provides detailed analysis
# 8. Carlos shares processed information with customer
```

### Demo Script Suggestions

#### 1. **Initial Contact**
```
"Hi, I'm looking for a car"
```
*Expected: Carlos greets and builds rapport*

#### 2. **Family Needs**
```
"I need a bigger, safer car because we just had a baby"
```
*Expected: Carlos updates customer profile, shows understanding*

#### 3. **Specific Requirements**
```
"I want a red sedan that's less than 2 years old"
```
*Expected: Carlos consults Manager, searches inventory*

#### 4. **Brand Preference**
```
"I'm interested in BMW vehicles"
```
*Expected: Carlos refines search, presents BMW options*

#### 5. **Technical Inquiry**
```
"What safety features does it have for babies?"
```
*Expected: Carlos requests research from María*

#### 6. **Specific Details**
```
"What's the trunk space in the BMW X3?"
```
*Expected: María provides technical specifications*

#### 7. **Pricing**
```
"What's the price of the black BMW X3?"
```
*Expected: Carlos consults Manager for official pricing*

#### 8. **Negotiation**
```
"Can you offer any discount?"
```
*Expected: Manager evaluates policies, Carlos negotiates*

#### 9. **Purchase Decision**
```
"I'll take it"
```
*Expected: Carlos finalizes sale, updates inventory*

---

## 🏗️ Architecture

### System Components

```mermaid
graph TB
    Client[👤 Customer]
    
    subgraph "Multi-Agent System"
        Carlos[🎯 Carlos<br/>Sales Agent<br/>GPT-4o]
        Maria[🔬 María<br/>Research<br/>o4-mini]
        Manager[🏢 Manager<br/>Coordinator<br/>GPT-4o]
        
        subgraph "Carlos Tools"
            T1[ConsultManager]
            T2[ResearchVehicleInfo]
            T3[UpdateCustomerProfile]
            T4[UpdateSalesStage]
            T5[RespondToClient]
            T6[FinalizeSale]
            T7[UpdateNotes]
        end
    end
    
    subgraph "External Systems"
        Inventory[🚗 Inventory DB]
        SerpAPI[🌐 SerpAPI]
        KB[📚 Knowledge Base]
    end
    
    Client <--> Carlos
    Carlos <--> T1
    Carlos <--> T2
    T1 <--> Manager
    T2 <--> Maria
    Manager <--> Inventory
    Maria <--> SerpAPI
    Maria <--> KB
```

---

## 🛠️ Technical Stack

### Core Technologies
- **🤖 AI Models**: OpenAI GPT-4o, o4-mini
- **🔗 Agent Framework**: LangChain
- **🖥️ Frontend**: Streamlit
- **📊 Data Processing**: Pandas, NumPy
- **📈 Visualization**: Plotly
- **🔍 Web Search**: SerpAPI
- **🐍 Language**: Python 3.8+

### Key Dependencies
```
langchain>=0.3.25
langchain-openai>=0.3.18
langchain-community>=0.3.24
streamlit>=1.45.1
pandas>=2.2.3
plotly>=5.17.0
openai>=1.82.0
python-dotenv>=1.1.0
```

---

## 📁 Project Structure

```
carbot-pro/
├── 📄 README.md                          # This file
├── 🚀 quick_setup_advanced.py            # Automated setup script
├── 🎯 advanced_multi_agent_system.py     # Core multi-agent system
├── 📦 enhanced_inventory_manager.py      # Inventory management
├── 🖥️ enhanced_app.py                    # Streamlit interface
├── 🧪 test_system.py                     # System tests
├── 📋 requirements.txt                   # Python dependencies
├── ⚙️ config.env                         # Environment template
├── 📊 data/                              # Vehicle database
└── 📝 carbot_system.log                  # System logs
```

---

## 🔧 Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ Yes | OpenAI API key for GPT-4o and o4-mini |
| `SERPAPI_API_KEY` | ❌ Optional | SerpAPI key for web research (fallback to knowledge base) |

### Agent Configuration

Each agent can be customized with different parameters:

```python
# Carlos (Sales Agent)
carlos_llm = ChatOpenAI(
    temperature=0.8,      # Creative for sales
    model_name="gpt-4o",  # Latest GPT-4o
    max_tokens=1000
)

# María (Research Agent)
maria_llm = ChatOpenAI(
    temperature=1,        # Factual for research
    model_name="o4-mini", # Efficient for analysis
    max_tokens=800
)

# Manager (Coordinator)
manager_llm = ChatOpenAI(
    temperature=0.4,      # Balanced for decisions
    model_name="gpt-4o",  # Strategic thinking
    max_tokens=600
)
```

---

## 📊 Analytics & Monitoring

### Available Metrics
- **Conversation Analytics**: Interaction count, agent communications
- **Sales Performance**: Stage progression, conversion tracking
- **Customer Profiling**: Profile completeness, preference analysis
- **Agent Efficiency**: Response times, tool usage patterns

### Logging System
The system provides comprehensive logging:
- **Agent Actions**: All agent decisions and tool usage
- **Inter-Agent Communications**: Message flow between agents
- **Customer Interactions**: Complete conversation history
- **System Events**: Inventory updates, errors, performance metrics

---

## 🧪 Testing

Run the test suite to verify system functionality:

```bash
python test_system.py
```

The test suite covers:
- ✅ Agent initialization
- ✅ Tool functionality
- ✅ Inventory operations
- ✅ Customer profile management
- ✅ Sales stage transitions

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Areas for Contribution
- 🔧 **New Agent Tools**: Expand Carlos's capabilities
- 🌐 **Additional Research Sources**: Integrate more data providers
- 📊 **Enhanced Analytics**: Advanced metrics and visualizations
- 🎨 **UI Improvements**: Better Streamlit interface
- 🧪 **Testing**: Expand test coverage
- 📚 **Documentation**: Improve guides and examples

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Add tests for new functionality
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**⭐ If you find this project useful, please consider giving it a star on GitHub!**

*Built with ❤️ for the AI community*
