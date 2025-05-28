import logging
import uvicorn
from fastapi import FastAPI
from routers import routers

from mangum import Mangum

app = FastAPI()


app.include_router(routers.router, prefix="/api")

handler = Mangum(app)

logging.basicConfig(level=logging.DEBUG)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
