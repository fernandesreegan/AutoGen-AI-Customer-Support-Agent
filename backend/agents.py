import os
import csv
from datetime import datetime
from backend.vector_store import KnowledgeBaseRAG

# Initialize our FAISS RAG instance
kb_rag = KnowledgeBaseRAG()

# ==========================================
# 1. DEFINE TOOLS FOR THE AGENTS
# ==========================================

def search_kb(query: str) -> str:
    """Searches the company knowledge base (shipping, refund, account, product FAQs) and returns matching passages."""
    results = kb_rag.search(query, k=2)
    if not results:
        return "No matching document found in the knowledge base."
    
    formatted_output = ""
    for r in results:
        formatted_output += f"Source: {r['source']}\nPassage: {r['passage']}\n\n"
    return formatted_output

def log_ticket(
    customer_name: str, 
    order_id: str, 
    complaint: str, 
    category: str, 
    urgency: str, 
    sentiment: str, 
    kb_sources: str, 
    response: str, 
    status: str, 
    escalation_reason: str
) -> str:
    """Logs the support ticket into tickets.csv with all required tracking details."""
    os.makedirs("logs", exist_ok=True)
    file_path = "logs/tickets.csv"
    
    file_exists = os.path.exists(file_path)
    
    # Generate unique ticket ID based on timestamp
    ticket_id = f"TKT-{datetime.now().strftime('%H%M%S')}"
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    row = [
        ticket_id, timestamp, f"{customer_name} {order_id}", complaint,
        category, urgency, sentiment, kb_sources, response, status, escalation_reason
    ]
    
    with open(file_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            # Write header row
            writer.writerow([
                "ticket_id", "timestamp", "customer_name / order_id", "complaint",
                "category", "urgency", "sentiment", "kb_sources", "response", "status", "escalation_reason"
            ])
        writer.writerow(row)
        
    return f"Ticket {ticket_id} successfully logged with status: {status}."


# ==========================================
# 2. SYSTEM MESSAGES FOR THE 4 AGENTS
# ==========================================

QUERY_AGENT_SYSTEM = """You are the Customer Query Agent. 
Your job is to read the raw customer complaint and extract:
- Category (e.g., Shipping, Refund, Account, Billing, Product)
- Urgency (Low, Medium, High)
- Sentiment (Neutral, Frustrated, Angry)
- Short Summary of the issue.
Pass this structured summary to the next agent."""

KNOWLEDGE_AGENT_SYSTEM = """You are the Knowledge Base Agent. 
Your job is to take the customer summary and category, and use the `search_kb` tool to find the exact company policy or FAQ passage that addresses the issue. Always output the matching passages along with their source file names."""

RESOLUTION_AGENT_SYSTEM = """You are the Resolution Agent. 
Your job is to write a clear, polite, and professional customer-ready reply that solves the problem. 
CRITICAL RULE: You must base your answer *only* on the facts retrieved by the Knowledge Base Agent. If no matching document was found or the facts are missing, honestly state that you cannot verify the policy and recommend escalation instead."""

LOG_AGENT_SYSTEM = """You are the Log Agent. 
Your job is to review the ticket details, evaluate escalation rules, and call the `log_ticket` tool.

ESCALATION RULES: Mark the ticket as 'Escalated' if:
1. Urgency is High
2. Customer sentiment is Angry or Frustrated
3. The Knowledge Base Agent found no matching document
4. The issue involves legal threats, fraud, or payment disputes
Otherwise, mark it as 'Resolved'.

CRITICAL INSTRUCTION: As soon as you call `log_ticket` and receive the success response, your final output message MUST contain the exact word "LOGGED" and nothing else."""