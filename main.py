import subprocess
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "abcd_fastapi_main:app",
        reload=True,
        port=8004,
    )
