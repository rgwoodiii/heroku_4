# Put the code for your API here.
import numpy as np
import pandas as pd
import os
from fastapi import FastAPI
from typing import Union, List, Optional


from pydantic import BaseModel, Field
import uvicorn
from joblib import load
import os.path
import sys



# instantiate app with fastapi
app = FastAPI()

    
# greeting
@app.get("/")
async def greet_user():
    return {"Welcome!"}

