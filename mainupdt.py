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

app = FastAPI(title="Client Chatbot")

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

# STATUS CONVERTER

def get_status_text(status):
    if status == 0:
        return "Not Started"
    elif status == 1:
        return "In Progress"
    elif status == 2:
        return "Completed"
    return "Unknown"

# USER → PROJECTS

@app.get("/users/{user_name}/projects")
async def get_user_projects(user_name: str):

    async with db_pool.acquire() as conn:

        user_query = 'SELECT id, name FROM "User" WHERE name ILIKE $1 LIMIT 1'
        user = await conn.fetchrow(user_query, f"%{user_name}%")

        if not user:
            return {"message": "User not found in the database.", "projects": []}

        #FIX: added status
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

# CHATBOT (OLLAMA)
class ChatRequest(BaseModel):
    user_name: str
    message: str

@app.post("/chat")
async def chat_with_bot(payload: ChatRequest):

    project_data = await get_user_projects(payload.user_name)

    projects = project_data.get("projects", [])

    if projects:
        project_summary = "\n".join(
            [f"{p['name']} (Status: {get_status_text(p['status'])})" for p in projects]
        )
    else:
        project_summary = "No projects found."

    # SYSTEM PROMPT
    system_instruction = f"""
    You are a helpful project assistant.

    User Info:
    {project_data.get('message')}

    Projects:
    {project_summary}

    Rules:
    - Answer clearly
    - Mention project status
    - If no user → say not found
    """

    try:
        # 🔥 OLLAMA CALL
        response = ollama.chat(
            model="llama3.2:1b",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": payload.message}
            ],
            options={"temperature": 0.2}
        )

        return {"reply": response["message"]["content"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)