from fastapi import FastAPI, BackgroundTasks, Form
from app.utils import retrieve_batch_results
from logger import api_logger as logger


app = FastAPI()


@app.post("/retrieve/{batch_id}")
def retrieve_data(
    batch_id: str,
    background_tasks: BackgroundTasks,
    email: str = Form(...),
):
    logger.info(f"Retrieval for batch {batch_id} started")
    logger.info(f"Recipient email: {email}")
    background_tasks.add_task(retrieve_batch_results, batch_id, email)
    return {"message": f"Retrieval for batch {batch_id} started"}
