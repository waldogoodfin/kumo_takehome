from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import search, products, dashboard
from app.state import get_data_bundle

settings = get_settings()

app = FastAPI(title="ShopSight API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    from app.state import init_data_bundle
    init_data_bundle()


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "ShopSight API is running!"}


@app.post("/refresh-data")
def refresh_data() -> dict[str, str]:
    from app.state import refresh_transactions
    refresh_transactions()
    return {"message": "Transaction data refreshed!"}


app.include_router(search.router)
app.include_router(products.router)
app.include_router(dashboard.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
