import logging
import pickle
from pathlib import Path
from typing import Optional
import httpx
import aiofiles
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from langchain.vectorstores import VectorStore
from callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from query_data import get_chain
from schemas import ChatResponse
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")
vectorstore: Optional[VectorStore] = None


def merge_pkl_files(input_path, output_path):
    merged_data = []

    # Iterate through all files in the input_path
    for filename in os.listdir(input_path):
        if filename.endswith(".pkl"):
            # Load data from the pickle file
            with open(os.path.join(input_path, filename), 'rb') as f:
                data = pickle.load(f)
                merged_data.extend(data)

    # Save the merged data to the output_path
    with open(output_path, 'wb') as f:
        pickle.dump(merged_data, f)



@app.on_event("startup")
async def startup_event():
    logging.info("loading vectorstore")
    if not Path("vectorstore.pkl").exists():
        # raise ValueError("vectorstore.pkl does not exist, please run ingest.py first")
        input_path = "vector_drive"
        output_path = "vectorstore.pkl"
        merge_pkl_files(input_path, output_path)
    with open("vectorstore.pkl", "rb") as f:
        global vectorstore
        vectorstore = pickle.load(f)

@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    question_handler = QuestionGenCallbackHandler(websocket)
    stream_handler = StreamingLLMCallbackHandler(websocket)
    chat_history = []
    qa_chain = get_chain(vectorstore, question_handler, stream_handler, tracing=True)

    while True:
        try:
            question = await websocket.receive_text()
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())

            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())

            result = await qa_chain.acall(
                {"question": question, "chat_history": chat_history}
            )
            chat_history.append((question, result["answer"]))

            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())
class SavePdfRequest(BaseModel):
    url: str
    name: str

@app.post("/save_pdf")
async def save_pdf(request: SavePdfRequest):
    url = request.url
    name = request.name
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            # find the filename from the response
            
        if response.status_code == 200:
            async with aiofiles.open(f'downloads/{name}.pdf', 'wb') as f:
                await f.write(response.content)
        else:
            raise HTTPException(status_code=400, detail=f"Error downloading PDF: Status code {response.status_code}")

    except Exception as e:
        return JSONResponse(content={'success': False, 'message': 'Error saving PDF: ' + str(e)}, status_code=400)

    return JSONResponse(content={'success': True, 'message': 'PDF saved successfully.'}, status_code=200)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9001)
