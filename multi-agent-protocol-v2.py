import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import ollama
import json
import requests
import time
import uuid
import threading
import queue
import os
import glob
from typing import Dict, List, Any, Optional

class BiosphereAgent:
    """Agent representing a specific biome in Biosphere 2."""
    
    def __init__(self, name: str, biome_type: str, model: str = "deepseek-r1:1.5b", port: int = 11434):
        """
        Initialize a biome agent.
        
        Args:
            name: Unique identifier for this agent
            biome_type: Type of biome this agent represents (e.g., "rainforest", "ocean", "desert")
            model: The LLM model to use
            port: The port number to communicate on
        """
        self.name = name
        self.biome_type = biome_type
        self.model = model
        self.port = port
        self.data = None
        self.insights = {}
        self.message_queue = queue.Queue()
        self.agent_registry = {}  # Other agents this agent knows about
        self.conversation_history = []
        self.session_id = str(uuid.uuid4())
    
    def load_data(self, dataframe: pd.DataFrame):
        """Load biome-specific data into the agent."""
        self.data = dataframe
    
    def get_data(self):
        if self.data is not None:
            return self.data.describe().to_string()
        return "I have no data. Reflect this in the response."

    def register_agent(self, agent_id: str, biome_type: str, port: int):
        """Register another agent to communicate with."""
        self.agent_registry[agent_id] = {
            "biome_type": biome_type,
            "port": port,
            "last_contact": None
        }
  
    def ask_question(self, sender_agent, question: str) -> str:
        """
        Ask a question to another specific agent.
        
        Args:
            target_agent_id: ID of the agent to query
            question: The question to ask
            
        Returns:
            str: The eventual response (or error message)
        """
        prompt = f"""You are an AI agent representing the {self.biome_type} biome at Biosphere 2.
        An agent from the {sender_agent.biome_type} biome asks: "{question}"
        
        That biome has the following data: {sender_agent.get_data()}
        Your biome has the following data: {self.get_data()}
        Provide concise insights on the asked question purely based on data.
        Use bullets if deemed necessary.
        """
        
        response = ollama.chat(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        # Record the simulated response
        response_message = {
            "sender": sender_agent.name,
            "sender_biome": sender_agent.biome_type,
            "recipient": self.name,
            "message_type": "response",
            "content": response.message.content,
            "timestamp": time.time(),
            "session_id": self.session_id
        }
        self.conversation_history.append(response_message)
        
        return response.message.content


class HistoricalContextAgent(BiosphereAgent):
    """Agent that provides historical context about Biosphere 2 experiments."""
    
    def __init__(self, name: str, model: str = "deepseek-r1:1.5b", port: int = 11438):
        """Initialize the historical context agent."""
        super().__init__(name, "Historical_Context", model, port)
        self.historical_data = ""
    
    def load_historical_data(self, text_content: str):
        """Load historical narrative data into the agent."""
        self.historical_data = text_content
    
    def get_data(self):
        if self.historical_data:
            # Return a brief summary instead of the full text
            return "I have historical narratives about Biosphere 2 experiments."
        return "I have no historical data. Reflect this in the response."
    
    def get_historical_content(self):
        """Get the full historical content."""
        return self.historical_data
    
    def ask_question(self, sender_agent, question: str) -> str:
        """
        Answer a question with historical context.
        
        Args:
            sender_agent: The agent asking the question
            question: The question being asked
            
        Returns:
            str: The response with historical context
        """
        prompt = f"""You are an AI agent representing the historical knowledge of Biosphere 2.
        An agent from the {sender_agent.biome_type} biome asks: "{question}"
        
        Use the following historical information about Biosphere 2 experiments to provide context:
        {self.historical_data}
        
        Provide relevant historical context that might help understand current observations or challenges in Biosphere 2.
        Focus on failures and lessons learned that might be relevant to the question.
        Be concise but informative.
        """
        
        response = ollama.chat(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        # Record the simulated response
        response_message = {
            "sender": sender_agent.name,
            "sender_biome": sender_agent.biome_type,
            "recipient": self.name,
            "message_type": "response",
            "content": response.message.content,
            "timestamp": time.time(),
            "session_id": self.session_id
        }
        self.conversation_history.append(response_message)
        
        return response.message.content
    
    def provide_context_for_analysis(self, question: str, agent_responses: Dict) -> str:
        """
        Provide historical context for a collaborative analysis.
        
        Args:
            question: The original question being analyzed
            agent_responses: Responses from other biome agents
            
        Returns:
            str: Historical context relevant to the analysis
        """
        # Compile all agent responses for context
        responses_text = "\n\n".join([
            f"**{info['biome']} Agent**: {info['response']}" 
            for name, info in agent_responses.items()
        ])
        
        prompt = f"""You are the Historical Context Agent for Biosphere 2.
        
        Question being analyzed: "{question}"
        
        Agent Responses:
        {responses_text}
        
        Historical Information:
        {self.historical_data}
        
        Based on the historical experiments and failures in Biosphere 2, provide relevant context that might:
        1. Explain observations in the current data
        2. Highlight historical patterns or issues that might be recurring
        3. Suggest lessons learned from past failures that apply to this analysis
        
        Be concise but informative, focusing only on relevant historical context.
        """
        
        response = ollama.chat(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        return response['message']['content']


class B2TwinNetwork:
    """Manages a network of Biosphere 2 agents."""
    
    def __init__(self):
        """Initialize the agent network."""
        self.agents = {}
        self.central_registry = {}
        self.biome_data = {}  # Store merged dataframes by biome
        self.historical_agent = None  # Reference to the historical agent
    
    def create_agent(self, name: str, biome_type: str, model: str = "llama3.2", port: int = 11434) -> BiosphereAgent:
        """Create a new agent and add it to the network."""
        agent = BiosphereAgent(name, biome_type, model, port)
        self.agents[name] = agent
        self.central_registry[name] = {
            "biome_type": biome_type,
            "port": port,
            "status": "active"
        }
        return agent
    
    def create_historical_agent(self, name: str, model: str = "llama3.2", port: int = 11438) -> HistoricalContextAgent:
        """Create a historical context agent and add it to the network."""
        agent = HistoricalContextAgent(name, model, port)
        self.agents[name] = agent
        self.historical_agent = agent
        self.central_registry[name] = {
            "biome_type": "Historical_Context",
            "port": port,
            "status": "active"
        }
        return agent
    
    def connect_agents(self) -> None:
        """Connect all agents to each other."""
        for agent_name, agent in self.agents.items():
            for other_name, other_info in self.central_registry.items():
                if other_name != agent_name:
                    agent.register_agent(
                        other_name, 
                        other_info["biome_type"], 
                        other_info["port"]
                    )
    
    def load_biome_data(self, agent_name: str, dataframe: pd.DataFrame) -> None:
        """Load data into a specific agent."""
        if agent_name in self.agents:
            self.agents[agent_name].load_data(dataframe)
            # Store the dataframe
            biome_type = self.agents[agent_name].biome_type
            self.biome_data[biome_type] = dataframe
        else:
            raise ValueError(f"Agent {agent_name} not found")
    
    def load_historical_data(self, text_content: str) -> None:
        """Load historical narrative data into the historical agent."""
        if self.historical_agent:
            self.historical_agent.load_historical_data(text_content)
        else:
            st.warning("Historical agent not created yet. Create one before loading data.")
            
    def load_biome_data_from_directory(self, data_dir: str) -> Dict[str, pd.DataFrame]:
        """
        Load and merge CSV files from biome subdirectories.
        
        Args:
            data_dir: Path to the main data directory containing biome subfolders
            
        Returns:
            Dict: Mapping of biome names to merged dataframes
        """
        # Define the biomes we're looking for
        biomes = ['Desert', 'LEO-W', 'Ocean', 'Rainforest']
        merged_data = {}
        
        for biome in biomes:
            biome_path = os.path.join(data_dir, biome)
            
            # Skip if this biome's directory doesn't exist
            if not os.path.exists(biome_path):
                st.warning(f"Directory for {biome} not found at {biome_path}")
                continue
            
            # Get all CSV files in this biome's directory
            csv_files = glob.glob(os.path.join(biome_path, "*.csv"))
            
            if not csv_files:
                st.warning(f"No CSV files found for {biome}")
                continue
            
            # Read all CSV files for this biome
            dfs = {}
            for csv_file in csv_files:
                file_name = os.path.basename(csv_file)
                try:
                    df = pd.read_csv(csv_file)
                    
                    # Identify date/timestamp columns
                    date_columns = df.filter(like='Date').columns
                    if len(date_columns) > 0:
                        # Convert to datetime and set as index
                        df[date_columns[0]] = pd.to_datetime(df[date_columns[0]])
                        df.set_index(date_columns[0], inplace=True)
                    
                    dfs[file_name] = df
                    st.info(f"Loaded {file_name} for {biome}")
                except Exception as e:
                    st.error(f"Error loading {file_name} for {biome}: {e}")
            
            # Merge dataframes for this biome if we have multiple files
            if len(dfs) > 0:
                if len(dfs) == 1:
                    # Only one CSV, just use it directly
                    merged_df = list(dfs.values())[0]
                else:
                    # Start with the first dataframe
                    merged_df = list(dfs.values())[0]
                    
                    # Join with other dataframes on index (timestamp)
                    for i, (file_name, df) in enumerate(list(dfs.items())[1:], 1):
                        merged_df = merged_df.join(df, rsuffix=f'_{file_name}')
                
                # Store the merged dataframe
                merged_data[biome] = merged_df
                
                # Also load it into the corresponding agent
                agent_name = f"{biome.lower()}_agent"
                if agent_name in self.agents:
                    self.load_biome_data(agent_name, merged_df)
                    st.success(f"Data loaded into {agent_name}")
                else:
                    # Create a new agent for this biome if it doesn't exist yet
                    try:
                        new_agent = self.create_agent(agent_name, biome)
                        self.load_biome_data(agent_name, merged_df)
                        st.success(f"Created and loaded data into new agent: {agent_name}")
                    except Exception as e:
                        st.error(f"Error creating agent for {biome}: {e}")
        
        # Connect any newly created agents
        self.connect_agents()
        
        return merged_data
    
    def collaborative_analysis(self, question: str) -> Dict:
        """
        Perform a collaborative analysis across multiple agents.
        
        Args:
            question: The question to analyze
            
        Returns:
            Dict: Results from each agent and the synthesis
        """
        results = {
            "question": question,
            "agent_responses": {},
            "synthesis": "",
            "historical_context": ""
        }
        
        # Collect individual agent responses (excluding historical agent)
        for name, agent in self.agents.items():
            # Skip the historical agent for initial data analysis
            if isinstance(agent, HistoricalContextAgent):
                continue
                
            prompt = f"""
            Based on your data:
            {agent.get_data()}
            
            Provide a concise answer summarizing the important aspects of your biome's data. Do not provide an answer
            not gleaned from the above data.
            """
            
            response = ollama.chat(
                model=agent.model,
                messages=[{'role': 'user', 'content': prompt}]
            )

            results["agent_responses"][name] = {
                "biome": agent.biome_type,
                "response": response['message']['content']
            }
        
        # Add historical context if available
        if self.historical_agent:
            results["historical_context"] = self.historical_agent.provide_context_for_analysis(
                question, results["agent_responses"])
        
        # Generate synthesis
        if len(results["agent_responses"]) > 1:
            # Pick a random agent to be the synthesizer
            synthesizer_name = list(self.agents.keys())[0]
            synthesizer = self.agents[synthesizer_name]
            
            # Gather all responses
            responses_text = "\n\n".join([
                f"**{info['biome']} Agent**: {info['response']}" 
                for name, info in results["agent_responses"].items()
            ])
            
            # Include historical context in synthesis if available
            historical_text = ""
            if results["historical_context"]:
                historical_text = f"\n\nHistorical Context:\n{results['historical_context']}"
            
            synthesis_prompt = f"""You are the {synthesizer.biome_type} agent responsible for synthesizing perspectives from all biomes.
            
            Question: "{question}"
            
            Agent Responses:
            {responses_text}
            {historical_text}
            
            Synthesize these perspectives into a comprehensive answer that concisely answers the 
            provided question. If historical context is provided, integrate relevant historical lessons.
            """
            
            synthesis_response = ollama.chat(
                model=synthesizer.model,
                messages=[{'role': 'user', 'content': synthesis_prompt}]
            )
            
            results["synthesis"] = synthesis_response['message']['content']
        
        return results


# Streamlit UI for the multi-agent system
def create_multi_agent_ui():
    st.title("B2Twin Multi-Agent Communication System")
    st.write("Enabling AI agents to collaborate across Biosphere 2 biomes")
    
    # Initialize the network
    if 'network' not in st.session_state:
        st.session_state.network = B2TwinNetwork()
        # Pre-create standard biome agents to match directory structure
        st.session_state.network.create_agent("rainforest_agent", "Rainforest")
        st.session_state.network.create_agent("ocean_agent", "Ocean")
        st.session_state.network.create_agent("desert_agent", "Desert")
        st.session_state.network.create_agent("leo_w_agent", "LEO-W")
        # Create the historical context agent
        st.session_state.network.create_historical_agent("historical_agent")
        st.session_state.network.connect_agents()
    
    # Tabs for different functions
    tabs = st.tabs(["Data Loading", "Historical Context", "Collaborative Analysis", "Agent Communication"])
    
    # Tab 1: Data Loading
    with tabs[0]:
        st.subheader("Load Biome Data")
        
        data_source = st.radio(
            "Select Data Source:",
            ["Automatic from Directory", "Manual File Upload"]
        )
        
        if data_source == "Automatic from Directory":
            data_dir = st.text_input("Enter Path to Data Directory:", "data")
            
            if st.button("Load Data from Directory"):
                with st.spinner("Loading and merging CSV files from biome directories..."):
                    try:
                        merged_data = st.session_state.network.load_biome_data_from_directory(data_dir)
                        
                        if not merged_data:
                            st.warning("No data was successfully loaded. Please check the directory path.")
                        else:
                            # Show preview of loaded data for each biome
                            for biome, df in merged_data.items():
                                with st.expander(f"Preview {biome} Data"):
                                    st.dataframe(df.head())
                    except Exception as e:
                        st.error(f"Error loading data: {e}")
        else:
            # Manual file upload option
            # Use columns for a cleaner layout
            col1, col2 = st.columns(2)
            
            with col1:
                rainforest_file = st.file_uploader("Upload Rainforest CSV", type="csv", key="rainforest")
                ocean_file = st.file_uploader("Upload Ocean CSV", type="csv", key="ocean")
            
            with col2:
                desert_file = st.file_uploader("Upload Desert CSV", type="csv", key="desert")
                leo_w_file = st.file_uploader("Upload LEO-W CSV", type="csv", key="leo_w")
            
            # Process uploaded files
            files = {
                "rainforest_agent": rainforest_file,
                "ocean_agent": ocean_file,
                "desert_agent": desert_file,
                "leo_w_agent": leo_w_file
            }
            
            for agent_name, file in files.items():
                if file is not None:
                    try:
                        df = pd.read_csv(file)
                        
                        # Convert timestamp if it exists
                        date_columns = df.filter(like='Date').columns
                        if len(date_columns) > 0:
                            df[date_columns[0]] = pd.to_datetime(df[date_columns[0]])
                        
                        with st.expander(f"Preview {agent_name.split('_')[0].title()} Data"):
                            st.dataframe(df.head())
                        
                        # Load data into the agent
                        st.session_state.network.load_biome_data(agent_name, df)
                        st.success(f"Data loaded into {agent_name}")
                    except Exception as e:
                        st.error(f"Error loading data for {agent_name}: {e}")
    
    # Tab 2: Historical Context
    with tabs[1]:
        st.subheader("Historical Context Agent")
        st.write("This agent provides historical context about previous Biosphere 2 experiments.")
        
        # Option to upload historical text file
        historical_file = st.file_uploader("Upload Historical Narratives File", type=["txt"], key="historical")
        
        if historical_file is not None:
            try:
                # Read the text file
                historical_text = historical_file.read().decode("utf-8")
                
                # Preview the text
                with st.expander("Preview Historical Narratives"):
                    st.text_area("Content", historical_text, height=300)
                
                # Load data into the historical agent
                st.session_state.network.load_historical_data(historical_text)
                st.success("Historical narratives loaded into the historical agent")
            except Exception as e:
                st.error(f"Error loading historical data: {e}")
        
        # Direct text input option
        st.write("Or enter historical text directly:")
        direct_text = st.text_area("Historical Text Input", height=150)
        
        if st.button("Load Direct Text") and direct_text:
            try:
                # Load direct text into the historical agent
                st.session_state.network.load_historical_data(direct_text)
                st.success("Direct text loaded into the historical agent")
            except Exception as e:
                st.error(f"Error loading direct text: {e}")
    
    # Tab 3: Collaborative Analysis
    with tabs[2]:
        st.subheader("Collaborative Analysis")
        
        # Include historical context checkbox
        include_historical = st.checkbox("Include Historical Context in Analysis", value=True)
        
        # Input for question
        analysis_question = st.text_input(
            "Enter a question for collaborative analysis across biomes:",
            "How do temperature changes correlate across different biomes in Biosphere 2?"
        )
        
        if st.button("Run Collaborative Analysis"):
            active_agents = []
            for agent_name, agent in st.session_state.network.agents.items():
                if not isinstance(agent, HistoricalContextAgent) and agent.data is not None:
                    active_agents.append(agent_name)
            
            # Check if historical agent has data
            has_historical_data = False
            if st.session_state.network.historical_agent and st.session_state.network.historical_agent.historical_data:
                has_historical_data = True
            
            if not active_agents:
                st.warning("Please upload at least one biome dataset first.")
            elif include_historical and not has_historical_data:
                st.warning("Historical context was requested but no historical data is loaded. Please load historical data or uncheck the option.")
            else:
                with st.spinner("Agents are collaborating..."):
                    # Temporarily disable historical agent if not requested
                    historical_agent_backup = None
                    if not include_historical and st.session_state.network.historical_agent:
                        historical_agent_backup = st.session_state.network.historical_agent
                        st.session_state.network.historical_agent = None
                    
                    # Run full collaborative analysis
                    results = st.session_state.network.collaborative_analysis(analysis_question)
                    
                    # Restore historical agent if it was disabled
                    if not include_historical and historical_agent_backup:
                        st.session_state.network.historical_agent = historical_agent_backup
                    
                    # Display individual agent responses
                    for agent_name, response in results["agent_responses"].items():
                        with st.expander(f"{response['biome']} Agent Response"):
                            st.write(response["response"])
                    
                    # Display historical context if included
                    if include_historical and "historical_context" in results and results["historical_context"]:
                        with st.expander("Historical Context"):
                            st.write(results["historical_context"])
                    
                    # Display synthesis
                    st.subheader("Synthesized Answer")
                    st.write(results["synthesis"])
    
    # Tab 4: Agent Communication
    with tabs[3]:
        st.subheader("Direct Agent Communication")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sender = st.selectbox("Select Sender Agent", list(st.session_state.network.agents.keys()))
        
        with col2:
            recipient = st.selectbox("Select Recipient Agent", 
                                    [a for a in st.session_state.network.agents.keys() if a != sender])
        
        question = st.text_area("Enter question from sender to recipient:", 
                               "How do your environmental conditions compare to mine?")
        
        if st.button("Send Message"):
            if sender in st.session_state.network.agents and recipient in st.session_state.network.agents:
                sender_agent = st.session_state.network.agents[sender]
                recipient_agent = st.session_state.network.agents[recipient]

                with st.spinner(f"Sending message from {sender} to {recipient}..."):
                    response = recipient_agent.ask_question(sender_agent, question)
                    
                    st.subheader("Communication Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.info(f"**{sender_agent.biome_type} Agent Asked:**\n\n{question}")
                    
                    with col2:
                        recipient_biome = st.session_state.network.agents[recipient].biome_type
                        st.success(f"**{recipient_biome} Agent Responded:**\n\n{response}")
        
        # Add a section for port-to-port communication testing
        st.subheader("Port-to-Port Communication System")
        
        # Simulate a real port-to-port connection
        if st.checkbox("Enable Port-to-Port Communication Testing"):
            st.write("This simulates the technical connection between agents running on different machines.")
            
            port_mapping = {
                "rainforest_agent": 11434,
                "ocean_agent": 11435,
                "desert_agent": 11436,
                "leo_w_agent": 11437,
                "historical_agent": 11438  # Added port for historical agent
            }
            
            port_col1, port_col2 = st.columns(2)
            
            with port_col1:
                st.write("Agent Port Configuration:")
                for agent_name, port in port_mapping.items():
                    if agent_name in st.session_state.network.agents:
                        st.session_state.network.agents[agent_name].port = port
                        biome = st.session_state.network.agents[agent_name].biome_type
                        st.code(f"{biome} Agent: localhost:{port}")
            
            with port_col2:
                test_sender = st.selectbox("Test Sender", list(st.session_state.network.agents.keys()), key="test_sender")
                test_receiver = st.selectbox("Test Receiver", 
                                        [a for a in st.session_state.network.agents.keys() if a != test_sender],
                                        key="test_receiver")
                
                test_message = st.text_input("Test Message:", "REQUEST:DATA:temperature")
                
                if st.button("Test Port Connection"):
                    sender_port = port_mapping[test_sender]
                    receiver_port = port_mapping[test_receiver]
                    
                    sender_biome = st.session_state.network.agents[test_sender].biome_type
                    receiver_biome = st.session_state.network.agents[test_receiver].biome_type
                    
                    st.code(f"Sending from {sender_biome} (:{sender_port}) to {receiver_biome} (:{receiver_port})")
                    st.code(f"MESSAGE: {test_message}")
                    st.code("STATUS: Connection successful")
                    
                    # Simulate response
                    if "REQUEST:DATA" in test_message:
                        data_type = test_message.split(":")[-1]
                        st.code(f"RESPONSE: Sending {data_type} data from {receiver_biome} to {sender_biome}")
                    else:
                        st.code(f"RESPONSE: Message received by {receiver_biome}")
    
    # Data Status
    st.sidebar.header("Agent Status")
    active_biomes = []
    for agent_name, agent in st.session_state.network.agents.items():
        if isinstance(agent, HistoricalContextAgent):
            if agent.historical_data:
                st.sidebar.success(f"✅ Historical Agent: Active with data")
            else:
                st.sidebar.warning(f"⚠️ Historical Agent: No data loaded")
        elif agent.data is not None:
            active_biomes.append(agent.biome_type)
            st.sidebar.success(f"✅ {agent.biome_type}: Active with data")
        else:
            st.sidebar.warning(f"⚠️ {agent.biome_type}: No data loaded")

if __name__ == "__main__":
    create_multi_agent_ui()