import uvicorn
from fastapi import FastAPI
from starlette.requests import Request

from ml_display.util.template_helpers import page_response

app = FastAPI(title="ml-display")


@app.get("/")
async def home(request: Request):
    return page_response("main.html", request)


def start():
    uvicorn.run("ml_display.main:app")


if __name__ == "__main__":
    start()
