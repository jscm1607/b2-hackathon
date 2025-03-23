# B2Twin: Multi-Agent Communication System for Biosphere 2

## Team - PROBEP
   - Diego Torres
   - Reika Fitzpatrick
   - Jose Santiago Campa Morales
   - Cynthia Eduwiges Navarro

## Overview

The B2Twin project is an innovative multi-agent AI system designed for Biosphere 2 research. It enables collaboration between AI agents representing different biomes within Biosphere 2, allowing for integrated data analysis, knowledge sharing, and complex ecosystem modeling. The system includes specialized agents for environmental data processing and historical context analysis, providing a comprehensive digital twin solution for Biosphere 2.

## Features

- **Multi-Agent Architecture**: Independent agents for Rainforest, Ocean, Desert, and LEO-W biomes plus a Historical Context agent
- **Agent Communication Protocol**: Inter-agent messaging system for knowledge sharing and collaborative analysis
- **Historical Context Integration**: Incorporates past experimental data and lessons learned from Biosphere 2 history
- **CSV Data Processing**: Upload and analyze real sensor data from Biosphere 2 biomes
- **Data Visualization**: Interactive charts, graphs, and statistics for environmental data analysis
- **Collaborative Analysis**: Cross-biome data synthesis for holistic ecosystem understanding

## Future Features
- **Math/Physicist-Agent**: This next agent can become crucial similarly to the historical context agent in helping biosphere researchers develop numerical models and simulations of derived insight. This Modeling Agent will be trained on math models related to the physics of the problem and/or propose ML/DL approaches for an optimized simulated environment/ for future experiments.
- **Checking Archive Results**: Expanding use of model 
- **Improving Communication Protocol**: Langgraph, Autogen. Better latency

## System Architecture

The B2Twin system consists of the following components:

1. **BiosphereAgent Class**: Base class for biome-specific agents
2. **HistoricalContextAgent Class**: Specialized agent for historical data integration
3. **B2TwinNetwork Class**: Manages the agent network and communication
4. **Streamlit UI**: User interface for interacting with the system

## Usage

1. **Run the application**:
   ```
   streamlit run multi-agent-v3.py
   ```

2. **Load Biome Data**:
   - Go to the "Data Loading" tab
   - Upload CSV files for each biome

3. **Load Historical Data**:
   - Go to the "Historical Context" tab
   - Upload a text file with historical narratives or enter text directly

4. **Run Collaborative Analysis**:
   - Go to the "Collaborative Analysis" tab
   - Enter your research question
   - Toggle historical context inclusion as needed
   - Review individual agent responses and the synthesized answer

5. **Test Agent Communication**:
   - Go to the "Agent Communication" tab
   - Select sender and recipient agents
   - Enter a question for direct agent-to-agent communication

6. **Visualize Data**:
   - Go to the "Data Visualization" tab
   - Upload CSV files (can be different from agent data)
   - Select visualization type and parameters
   - Explore data through interactive charts

## Port Configuration

The system uses the following default ports for agent communication:
- Rainforest Agent: localhost:11434
- Ocean Agent: localhost:11435
- Desert Agent: localhost:11436
- LEO-W Agent: localhost:11437
- Historical Agent: localhost:11438

## Example Queries

- "How do temperature changes correlate across different biomes in Biosphere 2?"
- "What impact does humidity have on plant growth in the Rainforest compared to the Desert?"
- "Based on historical experiments, what factors contributed to oxygen level fluctuations?"
- "How do current ocean pH levels compare to historical measurements?"

## LLM Integration

The system utilizes Ollama to run local Large Language Models (LLMs). The default model is "deepseek-r1:1.5b" but can be configured to use other compatible models based on your hardware capabilities.

## Development and Extension

The B2Twin system is designed to be modular and extensible. New agent types can be added by extending the BiosphereAgent class, and additional data visualization components can be integrated into the Streamlit UI.

## Hackathon Context

This project was developed during the B2Twin Hackathon (March 22-23, 2025). The goal was to create a digital twin of Biosphere 2 using AI to help scientists restore degraded environments on Earth and prepare for space travel.

## Contributors

Developed by participants in the B2Twin Hackathon, with support from AI Core and Biosphere 2 research teams.
