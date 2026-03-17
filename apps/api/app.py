from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from training.knn_model_v3 import OrganizationCategorizerV3

app = FastAPI()

origins = [
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = OrganizationCategorizerV3()
model.load_model("model/org_categorizer_knn_model_v3.pkl")

class CompanyRequest(BaseModel):
    company: str

@app.post("/predict")
def predict_company(req: CompanyRequest):
    org_name = req.company
    pred = model.predict(org_name, 0.47)

    return {
        "industry": pred["Industry"],
        "group": pred["Group"],
        "business": pred["Business"],
        "similar_orgs": pred["similar_orgs"],
        "confidence": pred.get("confidence", 0.0)
    }