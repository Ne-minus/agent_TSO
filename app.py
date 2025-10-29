import uvicorn
from fastapi import FastAPI

from api.v1.endpoints import router
from logger import LOGGING_CONFIG

app = FastAPI()

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, log_config=LOGGING_CONFIG, log_level="debug")
