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

class B2TwinNetwork:
    """Manages a network of Biosphere 2 agents."""
    
    def __init__(self):
        """Initialize the agent network."""
        self.agents = {}
        self.central_registry = {}
        self.biome_data = {}  # Store merged dataframes by biome
    
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
            "synthesis": ""
        }
        
        # Collect individual agent responses
        for name, agent in self.agents.items():
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
            
            synthesis_prompt = f"""You are the {synthesizer.biome_type} agent responsible for synthesizing perspectives from all biomes.
            
            Question: "{question}"
            
            Agent Responses:
            {responses_text}
            
            Synthesize these perspectives into a comprehensive answer that concisely answers the 
            provided question.
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
        st.session_state.network.connect_agents()
    
    # Data loading options
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
    
    # Collaborative Analysis section
    st.subheader("Collaborative Analysis")
    
    # Input for question
    analysis_question = st.text_input(
        "Enter a question for collaborative analysis across biomes:",
        "How do temperature changes correlate across different biomes in Biosphere 2?"
    )
    
    if st.button("Run Collaborative Analysis"):
        if not any(files.values()):
            st.warning("Please upload at least one biome dataset first.")
        else:
            with st.spinner("Agents are collaborating..."):
                # Only include agents that have data
                active_agents = []
                for agent_name, file in files.items():
                    if file is not None:
                        active_agents.append(agent_name)
                
                if len(active_agents) > 1:
                    # Run full collaborative analysis
                    results = st.session_state.network.collaborative_analysis(analysis_question)
                    
                    # Display individual agent responses
                    for agent_name, response in results["agent_responses"].items():
                        with st.expander(f"{response['biome']} Agent Response"):
                            st.write(response["response"])
                    
                    # Display synthesis
                    st.subheader("Synthesized Answer")
                    st.write(results["synthesis"])
                elif len(active_agents) == 1:
                    # Just get response from the single agent
                    agent = st.session_state.network.agents[active_agents[0]]
                    prompt = f"""Based on your {agent.biome_type} data, answer this question:
                    "{analysis_question}"
                    """
                    response = ollama.chat(
                        model=agent.model,
                        messages=[{'role': 'user', 'content': prompt}]
                    )
                    st.subheader(f"{agent.biome_type} Agent Response")
                    st.write(response['message']['content'])
    
    # Data Status
    active_biomes = []
    for agent_name, agent in st.session_state.network.agents.items():
        if agent.data is not None:
            active_biomes.append(agent.biome_type)
    
    if active_biomes:
        st.success(f"Active biomes with data: {', '.join(active_biomes)}")
    else:
        st.warning("No biomes have data loaded yet. Please load data to enable collaboration.")
    
    # Agent-to-Agent Communication
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
            "leo_w_agent": 11437
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

if __name__ == "__main__":
    create_multi_agent_ui()
