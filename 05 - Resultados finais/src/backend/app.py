from fastapi import FastAPI
from routes import Routes
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.include_router(Routes().router)

app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"]
    )

backend_dir = Path(__file__).resolve().parent
frontend_dist = (backend_dir.parent / "frontend" / "dist").resolve()

app.mount(
    "/static",
    StaticFiles(directory=str(frontend_dist), html=True),
    name="frontend"
)

@app.get("/")
def serve_frontend():
    return FileResponse(frontend_dist / "index.html")