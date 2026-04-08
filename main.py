import os
import asyncpg
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv
import uvicorn

# 1. Bootstrapping & Config
load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")
db_uri = os.getenv("DB_URI")

if not gemini_key or not db_uri:
    raise ValueError("Missing GEMINI_API_KEY or DB_URI in the .env file!")

ai_client = genai.Client(api_key=gemini_key)
app = FastAPI(title="Client Chatbot ")

# Global database pool
db_pool = None

@app.on_event("startup")
async def startup_event():
    global db_pool
    db_pool = await asyncpg.create_pool(db_uri)

@app.on_event("shutdown")
async def shutdown_event():
    if db_pool:
        await db_pool.close()


# ==========================================
# SCENARIO 1: The User-First Flow 
# ==========================================
@app.get("/users/{user_name}/projects")
async def get_user_projects(user_name: str):
    async with db_pool.acquire() as conn:
        # 1. Check if the user exists (Case-insensitive search using ILIKE)
        user_query = 'SELECT id, name FROM "User" WHERE name ILIKE $1 LIMIT 1'
        user = await conn.fetchrow(user_query, f"%{user_name}%")
        
        if not user:
            return {"message": "User not found in the database."}
        
        # 2. Fetch the projects they are assigned to
        project_query = """
            SELECT p.id, p.name 
            FROM "ProjectMember" pm
            JOIN "Project" p ON p.id = pm."projectId"
            WHERE pm."userId" = $1
        """
        projects = await conn.fetch(project_query, user["id"])
        
        # 3. Format the response exactly as discussed
        if projects:
            project_names = [p["name"] for p in projects]
            return {
                "user_id": user["id"],
                "message": f"{user['name']} is assigned to {len(projects)} project(s): {', '.join(project_names)}.",
                "projects": [dict(p) for p in projects]
            }
        else:
            return {
                "user_id": user["id"],
                "message": f"{user['name']} is not assigned to any projects.",
                "projects": []
            }


# ==========================================
# SCENARIO 2: The Project Roadmap Flow
# ==========================================
@app.get("/projects/{project_id}/roadmaps")
async def get_project_roadmaps(project_id: str):
    """
    Fetches a specific project and all its associated roadmaps.
    """
    async with db_pool.acquire() as conn:
        # 1. Verify project exists
        project = await conn.fetchrow('SELECT name, status FROM "Project" WHERE id = $1', project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found.")

        # 2. Extract roadmaps (FIXED: We now select ALL columns instead of guessing)
        roadmap_query = 'SELECT * FROM "Roadmap" WHERE "projectId" = $1'
        roadmaps = await conn.fetch(roadmap_query, project_id)

        # 3. We must convert datetime objects to strings so JSON can read them
        formatted_roadmaps = []
        for r in roadmaps:
            row_dict = dict(r)
            for key, value in row_dict.items():
                if hasattr(value, "isoformat"):  # Checks if it's a date/time
                    row_dict[key] = value.isoformat()
            formatted_roadmaps.append(row_dict)

        return {
            "project_name": project["name"],
            "total_roadmaps": len(formatted_roadmaps),
            "roadmaps": formatted_roadmaps
        }

# ==========================================
# SCENARIO 3: AI Integration (Putting it together)
# ==========================================
class ChatRequest(BaseModel):
    user_name: str
    message: str

@app.post("/chat")
async def chat_with_bot(payload: ChatRequest):
    """
    The main chatbot endpoint that uses the above logic to build context for Gemini.
    """
    # 1. Run Scenario 1 to get the user's project data
    project_data = await get_user_projects(payload.user_name)
    
    # 2. Build the guardrails for the AI
    system_instruction = f"""
    You are a helpful project management assistant. 
    The user is asking about: {payload.user_name}.
    
    [DATABASE CONTEXT]
    {project_data['message']}
    
    Rule 1: If the user has no projects, politely inform them.
    Rule 2: If the user is not found, state that clearly.
    Rule 3: Only list the projects provided in the context above.
    """
    
    try:
        # 3. Generate AI Response
        response = ai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=payload.message,
            config={"system_instruction": system_instruction, "temperature": 0.2}
        )
        return {"reply": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
