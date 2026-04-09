import os
import asyncpg
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama
from dotenv import load_dotenv
import uvicorn

# CONFIG
load_dotenv()
db_uri = os.getenv("DB_URI")

if not db_uri:
    raise ValueError("Missing DB_URI in .env file!")

app = FastAPI(title="Client Chatbot - Kanban & Llama Edition")

# DB pool
db_pool = None

@app.on_event("startup")
async def startup_event():
    global db_pool
    db_pool = await asyncpg.create_pool(db_uri)

@app.on_event("shutdown")
async def shutdown_event():
    if db_pool:
        await db_pool.close()

# 🔥 BUG FIX: ROBUST STATUS CONVERTER (Now Case-Insensitive)
def get_status_text(status):
    if not status: 
        return "Unknown"
    
    # Convert whatever comes from the DB into a lowercase string to prevent matching errors
    s = str(status).lower().strip()
    
    if s in ["0", "not started", "assigned", "todo"]:
        return "Not Started"
    elif s in ["1", "in progress", "doing", "active", "ongoing"]:
        return "In Progress"
    elif s in ["2", "completed", "done", "resolved"]:
        return "Completed"
    return "Unknown"

# ==========================================
# SCENARIO 1: USER → PROJECTS
# ==========================================
@app.get("/users/{user_name}/projects")
async def get_user_projects(user_name: str):
    async with db_pool.acquire() as conn:
        user_query = 'SELECT id, name FROM "User" WHERE name ILIKE $1 LIMIT 1'
        user = await conn.fetchrow(user_query, f"%{user_name}%")

        if not user:
            return {"message": "User not found in the database.", "projects": []}

        project_query = """
            SELECT p.id, p.name, p.status
            FROM "ProjectMember" pm
            JOIN "Project" p ON p.id = pm."projectId"
            WHERE pm."userId" = $1
        """
        projects = await conn.fetch(project_query, user["id"])

        return {
            "user_id": user["id"],
            "message": f"{user['name']} is assigned to {len(projects)} project(s).",
            "projects": [dict(p) for p in projects]
        }

# ==========================================
# SCENARIO 2: CHATBOT 
# ==========================================
class ChatRequest(BaseModel):
    project_id: str
    question: str

@app.post("/chat")
async def chat_with_bot(payload: ChatRequest):
    async with db_pool.acquire() as conn:
        # 1. Fetch the specific project
        project_query = 'SELECT id, name, status FROM "Project" WHERE id = $1'
        project = await conn.fetchrow(project_query, payload.project_id)
        
        if not project:
            raise HTTPException(status_code=404, detail="Project not found.")

        # 2. Fetch all tasks (features) for this specific project
        tasks = await conn.fetch('SELECT title, status FROM "Task" WHERE "projectId" = $1', payload.project_id)
        
        # 3. Sort tasks into Kanban columns
        kanban = {"Not Started": [], "In Progress": [], "Completed": [], "Unknown": []}
        for t in tasks:
            status_text = get_status_text(t['status'])
            if status_text in kanban:
                kanban[status_text].append(t['title'])
                
        # 4. Calculate Overall Project Progress Percentage
        total_tasks = len(tasks)
        completed_tasks = len(kanban["Completed"])
        progress_percent = int((completed_tasks / total_tasks * 100)) if total_tasks > 0 else 0
        
        # 5. Format the Kanban board for the AI
        task_str = f"  - Not Started (Assigned): {', '.join(kanban['Not Started']) or 'None'}\n"
        task_str += f"  - In Progress: {', '.join(kanban['In Progress']) or 'None'}\n"
        task_str += f"  - Completed: {', '.join(kanban['Completed']) or 'None'}"
        
        project_summary = f"Project: {project['name']}\nOverall Project Progress: {progress_percent}% Complete ({completed_tasks}/{total_tasks} tasks done)\n[Kanban Board Tasks]\n{task_str}"

    # 🔥 NEW: STRICTER SYSTEM PROMPT
    system_instruction = f"""
    You are a professional project manager AI reporting to a client. 
    
    Live Database Context:
    {project_summary}

    Rules for Answering "Status" or "Progress" questions:
    1. PROGRESS OVERVIEW: Briefly state the Overall Project Progress percentage.
    2. EXPLICIT LISTS: You MUST list the exact features/tasks based on their status using bullet points:
       - Say "These are the features completed:" and list the completed tasks.
       - Say "These features are ongoing:" and list the in-progress tasks.
       - Say "These features are not started:" and list the unassigned/todo tasks.
    3. SKIP EMPTY: If a category has 'None', do not list it. Only tell the user about the features that actually exist.
    """

    try:
        # OLLAMA CALL (Local Llama 3.2 1B)
        response = ollama.chat(
            model="llama3.2:1b",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": payload.question} 
            ],
            options={"temperature": 0.2}
        )

        return {"reply": response["message"]["content"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
