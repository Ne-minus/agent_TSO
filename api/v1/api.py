from fastapi import FastAPI
from api.v1.endpoints import router
from api.v1.small_endpoints import small_router


app = FastAPI()

app.include_router(router)
app.include_router(small_router)


@app.get("/")
async def root_endpoint():
    return {"status": "OK"}
