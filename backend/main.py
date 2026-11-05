from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import pandas as pd
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
import numpy as np

from backend.agents import (
    QUERY_AGENT_SYSTEM,
    KNOWLEDGE_AGENT_SYSTEM,
    RESOLUTION_AGENT_SYSTEM,
    LOG_AGENT_SYSTEM,
    search_kb,
    log_ticket
)

load_dotenv()

app = FastAPI(title="AI Customer Support Agent System")

class TicketRequest(BaseModel):
    customer_name: str = "Anonymous"
    order_id: str = "N/A"
    complaint: str

@app.post("/process-ticket")
async def process_ticket(req: TicketRequest):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is missing from .env file.")

    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=api_key)

    # Instantiate AutoGen agents
    query_agent = AssistantAgent(
        name="CustomerQueryAgent",
        model_client=model_client,
        system_message=QUERY_AGENT_SYSTEM
    )

    knowledge_agent = AssistantAgent(
        name="KnowledgeBaseAgent",
        model_client=model_client,
        system_message=KNOWLEDGE_AGENT_SYSTEM,
        tools=[search_kb]
    )

    resolution_agent = AssistantAgent(
        name="ResolutionAgent",
        model_client=model_client,
        system_message=RESOLUTION_AGENT_SYSTEM
    )

    log_agent = AssistantAgent(
        name="LogAgent",
        model_client=model_client,
        system_message=LOG_AGENT_SYSTEM,
        tools=[log_ticket]
    )

    # Define termination condition (stops when LOGGED is printed or max 10 messages)
    termination = TextMentionTermination("LOGGED") | MaxMessageTermination(max_messages=5)

    # Assemble team in sequential order: Query -> Knowledge -> Resolution -> Log
    team = RoundRobinGroupChat(
        participants=[query_agent, knowledge_agent, resolution_agent, log_agent],
        termination_condition=termination
    )

    # Initial prompt for the team
    task_prompt = f"Customer Name: {req.customer_name}\nOrder ID: {req.order_id}\nComplaint: {req.complaint}"

    agent_outputs = []
    async for message in team.run_stream(task=task_prompt):
        if hasattr(message, "content") and message.content:
            agent_outputs.append({
                "agent": getattr(message, "source", "System"),
                "content": str(message.content)
            })

    return {
        "status": "success",
        "agent_history": agent_outputs
    }

@app.get("/logs")
def get_logs():
    file_path = "logs/tickets.csv"
    if not os.path.exists(file_path):
        return {"logs": []}
    
    df = pd.read_csv(file_path)
    
    # Replace any NaN, null, or out-of-range float values with empty strings so JSON encoder doesn't crash
    df = df.replace({np.nan: "", None: ""})
    
    return {"logs": df.to_dict(orient="records")}