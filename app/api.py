"""FastAPI service. Loads models/model.joblib once at startup.

    uvicorn app.api:app --reload     then open http://127.0.0.1:8000/docs
"""
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src import config, predict

Month = Literal["Feb", "Mar", "May", "Jun", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Apr"]
VisitorType = Literal["Returning_Visitor", "New_Visitor", "Other"]


class Session(BaseModel):
    """The session as it looks at the moment of exit intent. PageValues is not asked for on purpose."""
    Administrative: int = Field(ge=0, le=100, description="Account / admin pages viewed so far")
    Administrative_Duration: float = Field(ge=0, le=86_400, description="Seconds on admin pages")
    Informational: int = Field(ge=0, le=100, description="Information pages (shipping, contact...) viewed")
    Informational_Duration: float = Field(ge=0, le=86_400, description="Seconds on information pages")
    ProductRelated: int = Field(ge=0, le=1000, description="Product pages viewed so far")
    ProductRelated_Duration: float = Field(ge=0, le=86_400, description="Seconds on product pages")
    BounceRates: float = Field(ge=0, le=1, description="Average GA bounce rate of the pages visited")
    ExitRates: float = Field(ge=0, le=1, description="Average GA exit rate of the pages visited")
    SpecialDay: float = Field(ge=0, le=1, description="Closeness to a holiday such as Valentine's Day, 0 to 1")
    # Jan and Apr never appear in the training data. They are accepted, and the pipeline treats them as unknown.
    Month: Month
    VisitorType: VisitorType
    OperatingSystems: int = Field(ge=1, le=20, description="Operating system code")
    Browser: int = Field(ge=1, le=20, description="Browser code")
    Region: int = Field(ge=1, le=9, description="Region code, 1 to 9")
    TrafficType: int = Field(ge=1, le=30, description="Traffic source code")
    Weekend: bool

    model_config = {"extra": "forbid", "json_schema_extra": {"examples": [predict.EXAMPLE]}}


class Prediction(BaseModel):
    purchase_probability: float
    show_popup: bool
    recommended_action: str
    threshold: float
    model_version: str


app = FastAPI(title="Discount pop-up targeting", version="1.0.0",
              description="Purchase probability for a browsing session at exit intent. Top 10% get the pop-up.")
model, meta = predict.load()  # loaded once at startup, not per request


@app.get("/health")
def health():
    return {"status": "ok", "model_version": meta["model_version"], "model_family": meta["model_family"],
            "mlflow_run_id": meta["mlflow_run_id"], "threshold": meta["threshold"]}


@app.post("/predict", response_model=Prediction)
def predict_one(session: Session):
    return predict.predict(session.model_dump())[0]
