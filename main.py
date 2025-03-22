import pandas as pd
import os

def load_biosphere_data(data_path):
    """
    Load Biosphere 2 sensor data from CSV files.
    
    Args:
        data_path: Path to directory containing CSV files
        
    Returns:
        Dictionary of DataFrames with sensor data
    """
    data_dict = {}
    
    for file in os.listdir(data_path):
        if file.endswith('.csv'):
            file_path = os.path.join(data_path, file)
            try:
                df = pd.read_csv(file_path)
                # Extract sensor name from filename
                sensor_name = os.path.splitext(file)[0]
                data_dict[sensor_name] = df
                print(f"Loaded {sensor_name} data with {len(df)} records")
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    return data_dict

import matplotlib.pyplot as plt

def visualize_sensor_data(data_dict, sensor_name, time_column='timestamp', value_column='value'):
    """
    Create a simple visualization of sensor data over time.
    
    Args:
        data_dict: Dictionary of sensor DataFrames
        sensor_name: Name of sensor to visualize
        time_column: Column name for timestamp
        value_column: Column name for sensor value
    """
    if sensor_name not in data_dict:
        print(f"Sensor {sensor_name} not found in data")
        return
    
    df = data_dict[sensor_name]
    
    plt.figure(figsize=(12, 6))
    plt.plot(df[time_column], df[value_column])
    plt.title(f"{sensor_name} Readings Over Time")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    import requests
import json

class OllamaWrapper:
    def __init__(self, model_name="llama2:7b", base_url="http://localhost:11434"):
        """
        Initialize Ollama LLM wrapper.
        
        Args:
            model_name: Name of the model to use
            base_url: URL for Ollama API
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_endpoint = f"{base_url}/api/generate"
        
    def query(self, prompt, system_prompt=None, temperature=0.7):
        """
        Send a query to the Ollama LLM.
        
        Args:
            prompt: User prompt to send to the model
            system_prompt: Optional system prompt for context
            temperature: Controls randomness (0.0-1.0)
            
        Returns:
            Model's response as a string
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "stream": False
        }
        
        if system_prompt:
            payload["system"] = system_prompt
            
        try:
            response = requests.post(self.api_endpoint, json=payload)
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            return f"Error querying LLM: {str(e)}"
        


from flask import Flask, request, jsonify
import threading

class AICommServer:
    def __init__(self, ollama_wrapper, port=5000):
        """
        Initialize communication server for AI agents.
        
        Args:
            ollama_wrapper: Instance of OllamaWrapper
            port: Port to run the server on
        """
        self.app = Flask(__name__)
        self.ollama = ollama_wrapper
        self.port = port
        self.setup_routes()
        
    def setup_routes(self):
        @self.app.route('/query', methods=['POST'])
        def handle_query():
            data = request.json
            if not data or 'prompt' not in data:
                return jsonify({"error": "Missing prompt in request"}), 400
                
            prompt = data['prompt']
            system_prompt = data.get('system_prompt')
            temperature = data.get('temperature', 0.7)
            
            response = self.ollama.query(prompt, system_prompt, temperature)
            return jsonify({"response": response})
            
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({"status": "healthy", "model": self.ollama.model_name})
    
    def start(self):
        """Start the server in a separate thread"""
        def run_server():
            self.app.run(host='0.0.0.0', port=self.port)
            
        server_thread = threading.Thread(target=run_server)
        server_thread.daemon = True
        server_thread.start()
        print(f"AI Communication Server running on port {self.port}")

import requests

class AICommClient:
    def __init__(self):
        """Initialize client for communicating with other AI agents"""
        self.known_agents = {}  # {agent_name: url}
        
    def register_agent(self, agent_name, url):
        """Register a new AI agent"""
        self.known_agents[agent_name] = url
        
    def query_agent(self, agent_name, prompt, system_prompt=None, temperature=0.7):
        """
        Send a query to another AI agent.
        
        Args:
            agent_name: Name of the agent to query
            prompt: Prompt to send
            system_prompt: Optional system context
            temperature: Controls randomness
            
        Returns:
            Agent's response or error message
        """
        if agent_name not in self.known_agents:
            return f"Unknown agent: {agent_name}"
            
        url = f"{self.known_agents[agent_name]}/query"
        payload = {
            "prompt": prompt,
            "temperature": temperature
        }
        
        if system_prompt:
            payload["system_prompt"] = system_prompt
            
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            return f"Error communicating with {agent_name}: {str(e)}"
        
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

def create_streamlit_app(data_loader, ollama_wrapper, comm_client):
    """
    Create a Streamlit app for the B2Twin project.
    
    Args:
        data_loader: Function to load data
        ollama_wrapper: Instance of OllamaWrapper
        comm_client: Instance of AICommClient
    """
    st.title("B2Twin: Biosphere 2 Digital Twin")
    
    # Sidebar for navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["Data Explorer", "AI Analysis", "Inter-AI Communication"]
    )
    
    if page == "Data Explorer":
        st.header("Biosphere 2 Sensor Data Explorer")
        
        # Data upload section
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write(f"Loaded data with {len(df)} records")
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Simple visualization
            st.subheader("Data Visualization")
            
            # Detect time and value columns
            time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
            time_col = st.selectbox("Select time column", time_cols if time_cols else df.columns)
            
            value_col = st.selectbox("Select value column", [col for col in df.columns if col != time_col])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df[time_col], df[value_col])
            ax.set_title(f"{value_col} over {time_col}")
            ax.set_xlabel(time_col)
            ax.set_ylabel(value_col)
            ax.grid(True)
            plt.xticks(rotation=45)
            st.pyplot(fig)
    
    elif page == "AI Analysis":
        st.header("AI-Powered Data Analysis")
        
        # Context input
        st.subheader("Provide Context")
        context = st.text_area(
            "Describe the data and what you want to analyze",
            "I have temperature and humidity data from Biosphere 2's rainforest biome. What insights can you provide?"
        )
        
        # Query LLM
        if st.button("Analyze with AI"):
            with st.spinner("AI is analyzing..."):
                system_prompt = """
                You are an expert environmental scientist specializing in closed ecosystem analysis.
                You help analyze data from Biosphere 2, a closed ecological system in Arizona.
                Provide scientific insights based on the data described.
                """
                
                response = ollama_wrapper.query(context, system_prompt)
                st.subheader("AI Analysis")
                st.write(response)
    
    elif page == "Inter-AI Communication":
        st.header("Communicate with Other AI Agents")
        
        # Agent registration
        st.subheader("Register New Agent")
        col1, col2 = st.columns(2)
        with col1:
            agent_name = st.text_input("Agent Name")
        with col2:
            agent_url = st.text_input("Agent URL (e.g., http://192.168.1.100:5000)")
        
        if st.button("Register Agent"):
            if agent_name and agent_url:
                comm_client.register_agent(agent_name, agent_url)
                st.success(f"Registered agent: {agent_name}")
            else:
                st.error("Please provide both agent name and URL")
        
        # Agent communication
        st.subheader("Query Agent")
        agent_to_query = st.selectbox("Select Agent", list(comm_client.known_agents.keys()) if comm_client.known_agents else ["No agents registered"])
        
        query = st.text_area("Your Query", "Can you analyze the rainforest temperature trends?")
        
        if st.button("Send Query"):
            if agent_to_query != "No agents registered":
                with st.spinner(f"Querying {agent_to_query}..."):
                    response = comm_client.query_agent(agent_to_query, query)
                    st.subheader("Response")
                    st.write(response)
            else:
                st.error("Please register at least one agent first")


def main():
    """Main entry point for the B2Twin application"""
    # Initialize components
    ollama = OllamaWrapper(model_name="llama2:7b")
    
    # Set up communication
    comm_client = AICommClient()
    comm_server = AICommServer(ollama, port=5000)
    comm_server.start()
    
    # Define data path
    data_path = "./data"  # Update with actual path to data
    
    # Create and run Streamlit app
    import streamlit
    streamlit.cli.main_run(["streamlit", "run", "app.py", "--", data_path])

if __name__ == "__main__":
    main()