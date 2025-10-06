import subprocess
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "abcd_admin_prompts_fastapi:app",
        reload=True,
        port=8001,
    )
    #subprocess.run(["uvicorn", "abcd_admin_prompts_fastapi:app", "--reload", "--port", "8001"])
    # subprocess.run(["nohup", "uvicorn", "abcd_admin_prompts_fastapi:app", "--reload", "--port", "8002", "--host", "0.0.0.0", "&"])
    
    
    