from fastapi import FastAPI
from api.v1.endpoints import router


app = FastAPI()

app.include_router(router)

@app.get("/")
async def root_endpoint():
    return {"status": "OK"}