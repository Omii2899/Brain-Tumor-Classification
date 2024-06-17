from fastapi import FastAPI
import pickle

app = FastAPI()

# @app.get('/')
# def read_root():
#     return{"hello":"World"}

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app.post('/')
async def prediction():
    