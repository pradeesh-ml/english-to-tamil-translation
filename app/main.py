
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from app.translator import translate
import uvicorn

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("app/templates/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/translate")
async def translate_text(text: str = Form(...)):
    translation, _ = translate(text)
    return {"translation": translation}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
