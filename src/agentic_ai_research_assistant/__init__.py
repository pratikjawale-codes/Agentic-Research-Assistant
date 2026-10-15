from agentic_ai_research_assistant.api.main import app

def main():
    import uvicorn
    uvicorn.run("agentic_ai_research_assistant.api.main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
