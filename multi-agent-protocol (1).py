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
    
    def __init__(self, name: str, biome_type: str, model: str = "llama3.2", port: int = 11434):
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
        # Generate basic statistics and observations about the data
        self._analyze_data()
    
    def _analyze_data(self):
        """Analyze loaded data and extract key insights."""
        if self.data is None:
            return
        
        # Basic statistical analysis
        stats = self.data.describe()
        
        # Generate insights based on the data
        prompt = f"""You are an AI agent representing the {self.biome_type} biome at Biosphere 2.
        Analyze this data summary and extract 3-5 key insights about the {self.biome_type} biome:
        
        {stats.to_string()}
        
        What are the most important patterns or characteristics of this biome based on the data?
        Format your response as a JSON object with insight categories as keys and descriptions as values.
        """
        
        response = ollama.chat(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        # Extract JSON from response
        try:
            # Try to parse the entire response as JSON
            self.insights = json.loads(response['message']['content'])
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from the text
            try:
                json_str = response['message']['content']
                # Find JSON block if it exists
                start = json_str.find('{')
                end = json_str.rfind('}') + 1
                if start >= 0 and end > start:
                    self.insights = json.loads(json_str[start:end])
                else:
                    # Create basic insights if JSON extraction fails
                    self.insights = {
                        "data_summary": "Statistical summary generated",
                        "error": "Could not parse AI-generated insights as JSON"
                    }
            except:
                self.insights = {"error": "Failed to generate insights"}
    
    def register_agent(self, agent_id: str, biome_type: str, port: int):
        """Register another agent to communicate with."""
        self.agent_registry[agent_id] = {
            "biome_type": biome_type,
            "port": port,
            "last_contact": None
        }
    
    def send_message(self, recipient_id: str, message_type: str, content: Any) -> bool:
        """
        Send a message to another agent.
        
        Args:
            recipient_id: ID of the recipient agent
            message_type: Type of message (query, response, insight, alert)
            content: Message content
            
        Returns:
            bool: Success status
        """
        if recipient_id not in self.agent_registry:
            print(f"Error: Unknown recipient {recipient_id}")
            return False
        
        recipient = self.agent_registry[recipient_id]
        
        message = {
            "sender": self.name,
            "sender_biome": self.biome_type,
            "recipient": recipient_id,
            "message_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "message_type": message_type,
            "content": content,
            "session_id": self.session_id
        }
        
        try:
            # In a real implementation, this would use actual network communication
            # For hackathon purposes, we're simulating with a local message queue
            # In production, this would be a POST request to the recipient's endpoint
            # requests.post(f"http://localhost:{recipient['port']}/message", json=message)
            
            # For now, we'll just print the message and add it to our history
            print(f"Message sent to {recipient_id}: {message_type}")
            self.conversation_history.append(message)
            return True
            
        except Exception as e:
            print(f"Error sending message: {e}")
            return False
    
    def receive_message(self, message: Dict) -> Dict:
        """
        Process an incoming message from another agent.
        
        Args:
            message: The incoming message
            
        Returns:
            Dict: Response message if any
        """
        # Validate the message
        required_fields = ["sender", "message_type", "content"]
        if not all(field in message for field in required_fields):
            return {"error": "Invalid message format"}
        
        # Add to history
        self.conversation_history.append(message)
        
        # Process based on message type
        if message["message_type"] == "query":
            # Generate a response to the query
            return self._process_query(message)
        elif message["message_type"] == "insight":
            # Store the insight from another biome
            return self._process_insight(message)
        elif message["message_type"] == "alert":
            # Handle an alert condition
            return self._process_alert(message)
        else:
            # Simply acknowledge receipt
            return {
                "sender": self.name,
                "recipient": message["sender"],
                "message_type": "acknowledgment",
                "content": f"Received {message['message_type']} message",
                "timestamp": time.time(),
                "session_id": self.session_id
            }
    
    def _process_query(self, message: Dict) -> Dict:
        """Process a query from another agent."""
        query = message["content"]
        
        # Generate response using the LLM
        prompt = f"""You are an AI agent representing the {self.biome_type} biome at Biosphere 2.
        Another agent from the {message['sender_biome']} biome asks: "{query}"
        
        Based on your data insights:
        {json.dumps(self.insights, indent=2)}
        
        How would you respond? Focus on how data from your biome relates to their question.
        """
        
        response = ollama.chat(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        return {
            "sender": self.name,
            "recipient": message["sender"],
            "message_type": "response",
            "content": response['message']['content'],
            "in_response_to": message.get("message_id"),
            "timestamp": time.time(),
            "session_id": self.session_id
        }
    
    def _process_insight(self, message: Dict) -> Dict:
        """Process an insight shared by another agent."""
        # Store the insight for future reference
        sender = message["sender"]
        if "external_insights" not in self.insights:
            self.insights["external_insights"] = {}
        if sender not in self.insights["external_insights"]:
            self.insights["external_insights"][sender] = []
        
        self.insights["external_insights"][sender].append(message["content"])
        
        return {
            "sender": self.name,
            "recipient": message["sender"],
            "message_type": "acknowledgment",
            "content": f"Thank you for sharing your insight about {message['sender_biome']}",
            "timestamp": time.time(),
            "session_id": self.session_id
        }
    
    def _process_alert(self, message: Dict) -> Dict:
        """Process an alert from another agent."""
        # For alerts, we might want to take some action and respond with our status
        alert = message["content"]
        
        # Generate response using the LLM
        prompt = f"""You are an AI agent representing the {self.biome_type} biome at Biosphere 2.
        Another agent from the {message['sender_biome']} has sent this alert: "{alert}"
        
        How might this alert affect your biome? What would you respond?
        """
        
        response = ollama.chat(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        return {
            "sender": self.name,
            "recipient": message["sender"],
            "message_type": "alert_response",
            "content": response['message']['content'],
            "timestamp": time.time(),
            "session_id": self.session_id
        }
    
    def generate_insight_to_share(self, target_agent_id: Optional[str] = None) -> Dict:
        """
        Generate a new insight to share with other agents.
        
        Args:
            target_agent_id: If specified, tailor the insight for this specific agent
            
        Returns:
            Dict: The message to send
        """
        target_biome = "all biomes" if target_agent_id is None else self.agent_registry[target_agent_id]["biome_type"]
        
        prompt = f"""You are an AI agent representing the {self.biome_type} biome at Biosphere 2.
        Generate one important insight from your data that would be valuable to share with {target_biome}.
        
        Your current insights are:
        {json.dumps(self.insights, indent=2)}
        
        Create a concise insight that connects your biome to others in the Biosphere 2 ecosystem.
        """
        
        response = ollama.chat(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        insight = response['message']['content']
        
        # If we have a specific target, send directly to them
        if target_agent_id is not None:
            self.send_message(target_agent_id, "insight", insight)
            return {"recipient": target_agent_id, "content": insight}
        
        # Otherwise, broadcast to all known agents
        for agent_id in self.agent_registry:
            self.send_message(agent_id, "insight", insight)
        
        return {"recipient": "broadcast", "content": insight}
    
    def ask_question(self, target_agent_id: str, question: str) -> str:
        """
        Ask a question to another specific agent.
        
        Args:
            target_agent_id: ID of the agent to query
            question: The question to ask
            
        Returns:
            str: The eventual response (or error message)
        """
        if target_agent_id not in self.agent_registry:
            return f"Error: Unknown agent {target_agent_id}"
        
        # Send the query
        success = self.send_message(target_agent_id, "query", question)
        if not success:
            return "Error: Failed to send message"
        
        # In a real implementation, we would wait for the response asynchronously
        # For hackathon purposes, we'll simulate it
        
        # Simulate the other agent's processing
        target_biome = self.agent_registry[target_agent_id]["biome_type"]
        
        prompt = f"""You are an AI agent representing the {target_biome} biome at Biosphere 2.
        An agent from the {self.biome_type} biome asks: "{question}"
        
        Generate a realistic response as if you had data about the {target_biome}.
        Focus on how your biome might interact with or differ from {self.biome_type}.
        """
        
        response = ollama.chat(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        simulated_response = response['message']['content']
        
        # Record the simulated response
        response_message = {
            "sender": target_agent_id,
            "sender_biome": target_biome,
            "recipient": self.name,
            "message_type": "response",
            "content": simulated_response,
            "timestamp": time.time(),
            "session_id": self.session_id
        }
        self.conversation_history.append(response_message)
        
        return simulated_response

    def collaborative_reasoning(self, question: str, agents: List[str]) -> str:
        """
        Engage in collaborative reasoning with multiple agents on a question.
        
        Args:
            question: The question or problem to reason about
            agents: List of agent IDs to collaborate with
            
        Returns:
            str: The final consensus or synthesis
        """
        # Initialize the reasoning process
        reasoning_thread = {
            "question": question,
            "contributions": [],
            "synthesis": ""
        }
        
        # Get initial perspective from this agent
        prompt = f"""You are an AI agent representing the {self.biome_type} biome at Biosphere 2.
        Initial question for collaborative reasoning: "{question}"
        
        Based on your insights and data, provide your initial perspective.
        Keep your response focused and under 150 words.
        """
        
        response = ollama.chat(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        # Add our contribution
        reasoning_thread["contributions"].append({
            "agent": self.name,
            "biome": self.biome_type,
            "content": response['message']['content'],
            "timestamp": time.time()
        })
        
        # Get contributions from other agents
        for agent_id in agents:
            if agent_id not in self.agent_registry:
                continue
                
            agent_biome = self.agent_registry[agent_id]["biome_type"]
            
            # Simulate the contribution from this agent
            prompt = f"""You are an AI agent representing the {agent_biome} biome at Biosphere 2.
            Question for collaborative reasoning: "{question}"
            
            The {self.biome_type} agent has already contributed:
            "{reasoning_thread['contributions'][0]['content']}"
            
            Provide your perspective as the {agent_biome} agent. How does your biome's data relate?
            Keep your response focused and under 150 words.
            """
            
            response = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}]
            )
            
            # Add this agent's contribution
            reasoning_thread["contributions"].append({
                "agent": agent_id,
                "biome": agent_biome,
                "content": response['message']['content'],
                "timestamp": time.time()
            })
        
        # Synthesize the collaborative reasoning
        contributions_text = "\n\n".join([
            f"**{c['biome']} Agent**: {c['content']}" 
            for c in reasoning_thread["contributions"]
        ])
        
        synthesis_prompt = f"""You are a scientific synthesis system at Biosphere 2.
        The following agents have contributed their perspectives on this question:
        "{question}"
        
        Contributions:
        {contributions_text}
        
        Synthesize these viewpoints into a cohesive answer that represents the collaborative intelligence
        of the Biosphere 2 ecosystem. Highlight points of consensus and identify any areas where
        different biomes provide complementary insights.
        """
        
        synthesis_response = ollama.chat(
            model=self.model,
            messages=[{'role': 'user', 'content': synthesis_prompt}]
        )
        
        reasoning_thread["synthesis"] = synthesis_response['message']['content']
        
        return reasoning_thread["synthesis"]


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
            prompt = f"""You are an AI agent representing the {agent.biome_type} biome at Biosphere 2.
            Analyze this question from your biome's perspective: "{question}"
            
            Based on your data insights:
            {json.dumps(agent.insights, indent=2)}
            
            Provide a concise answer focusing on your biome's unique contribution.
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
            
            Synthesize these perspectives into a comprehensive answer that leverages insights from
            all biomes. Highlight connections between biomes and how they collectively inform our
            understanding of Biosphere 2 as an integrated system.
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
            
            with st.spinner(f"Sending message from {sender} to {recipient}..."):
                response = sender_agent.ask_question(recipient, question)
                
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
