from fastapi import FastAPI, HTTPException, Cookie, Depends, status, APIRouter
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets
from fastapi import Request
from fastapi import Body
from pydantic import BaseModel
from firebase_admin import credentials, initialize_app, auth, exceptions
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from fastapi.responses import JSONResponse
from typing import List, Optional
from create_db import DB_User
import logging
import random

from fastapi import Cookie
from typing_extensions import Annotated
from typing import Union

logging.basicConfig(level=logging.INFO)

# Define the database URL
DATABASE_URL = "sqlite:///./test.db"

# Create an SQLAlchemy engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

sessions = {}

app = FastAPI()

# Firebase Configuration
firebase_cred = credentials.Certificate("smart-news-recommendation-firebase-adminsdk-i1z4c-cf83911ed0.json")
firebase_app = initialize_app(firebase_cred)

class GoogleLoginRequest(BaseModel):
    id_token: str

class UserRegistrationRequest(BaseModel):
    email: str = None  # Make email and password optional
    password: str = None
    id_token: str = None
    first_name: str
    last_name: str

class UserLoginRequest(BaseModel):
    email: str
    password: str

class User(BaseModel):
    first_name: str
    last_name: str
    email: str
    firebase_id: str

def get_all_users():
    with SessionLocal() as db:
        users = db.query(DB_User).all()
        for user in users:
            logging.info(f"User ID: {user.user_id}, Name: {user.first_name + ' ' + user.last_name}, Email: {user.email}, Firebase ID: {user.firebase_id}")


def get_user_by_firebase_id_db(firebase_id: str):
    with SessionLocal() as db:
        user = db.query(DB_User).filter(DB_User.firebase_id == firebase_id).first()
        return user

def create_session(user_id: int):
    session_id = len(sessions) + random.randint(0, 1000000)
    sessions[session_id] = user_id
    return session_id

from fastapi.responses import JSONResponse

@app.post("/login/google")
async def google_login(request_data: GoogleLoginRequest):
    try:
        # Verify the Google ID token
        decoded_token = auth.verify_id_token(request_data.id_token)
        get_all_users()
        # Get the user's UID
        uid = decoded_token['uid']
        logging.info('uid', uid)

        user = get_user_by_firebase_id_db(uid)
        session_id = create_session(user.user_id)

        # session = await session_manager.create_session()
        # session["user_token"] = uid

        # Return user information
        content = {
            "uid": uid,
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "session_id": session_id
        }
        response = JSONResponse(content=content)
        response.set_cookie(key="session_id", value=session_id)
        return response
    except exceptions.FirebaseError as e:
        raise HTTPException(status_code=401, detail="Invalid ID token")
    

# Custom middleware for session-based authentication
def get_authenticated_user_from_session_id(session_id : Annotated[Union[str, None], Cookie()] = None):
    logging.info('session_id', session_id)
    # logging.info('session_id2', request.cookies).get('session_id')
    session_id = session_id
    if session_id is None or int(session_id) not in sessions:
        raise HTTPException(
            status_code=401,
            detail="Invalid session ID",
        )
    # Get the user from the session
    user = get_user_from_session(int(session_id))
    return user


# Use the valid session id to get the corresponding user from the users dictionary
def get_user_from_session(session_id: int):
    user_id = sessions.get(session_id)
    user = get_user_db(user_id)

    return user


@app.get("/getusers/me")
def read_current_user(user: dict = Depends(get_authenticated_user_from_session_id)):
    return user

# Protected endpoint - Requires authentication
@app.get("/protected")
def protected_endpoint(user: dict = Depends(get_authenticated_user_from_session_id)):
    if user is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authenticated")
    return {"message": "This user can connect to a protected endpoint after successfully autheticated", "user": user}




def get_user_db(user_id: int):
    with SessionLocal() as db:
        user = db.query(DB_User).filter(DB_User.user_id == user_id).first()
        return user

def get_user_by_email_db(email: str):
    with SessionLocal() as db:
        user = db.query(DB_User).filter(DB_User.email == email).first()
        return user



def add_user_db(user: User):
    with SessionLocal() as db:
        user = DB_User(firebase_id=user.firebase_id, first_name=user.first_name, last_name=user.last_name, email=user.email)
        db.add(user)
        db.commit()

@app.post("/register/google")
async def google_register(request_data: UserRegistrationRequest):
    try:
        # Verify the Google ID token
        decoded_token = auth.verify_id_token(request_data.id_token)

        # Get the user's UID
        uid = decoded_token['uid']

        # Get the user's name from the request or any other source
        first_name = request_data.first_name
        last_name = request_data.last_name

        user = auth.get_user(uid)
        user = User(firebase_id=uid, first_name=first_name, last_name=last_name, email=user.email)
        
        add_user_db(user)
        

        # You can store user information in your database or perform other registration actions here
        # For demonstration purposes, we'll return user information
        return {
            "uid": uid,
            "email": user.email,
            'first_name': first_name,
            'last_name': last_name
        }
    except exceptions.FirebaseError as e:
        raise HTTPException(status_code=401, detail="Invalid ID token")
    
@app.post("/register/email")
async def email_register(request_data: UserRegistrationRequest):
    try:
        # Create a new user with email and password
        user = auth.create_user(
            email=request_data.email,
            password=request_data.password
        )

        user = User(firebase_id=user.uid, first_name=request_data.first_name, last_name=request_data.last_name, email=request_data.email)
        add_user_db(user)

        # You can store user information in your database or perform other registration actions here
        # For demonstration purposes, we'll return user information
        return {
            "uid": user.firebase_id,
            "email": user.email,
            'first_name': user.first_name,
            'last_name': user.last_name
        }
    except exceptions.FirebaseError as e:
        raise HTTPException(status_code=401, detail="Invalid ID token")

# @app.post("/login/email")
# async def email_login(request_data: UserLoginRequest):
#     # try:
#         # Sign in with email and password using Firebase Auth
#         user = auth.sign_in_with_email_and_password(request_data.email, request_data.password)

#         # You can access user information through the 'user' object
#         # For example, 'user['localId']' contains the user's unique identifier (UID)
#         return {
#             "uid": user['localId'],
#             "email": user['email'],
#             # You can add more user information here as needed
#         }
    # except exceptions.FirebaseAuthError as e:
    #     if isinstance(e, exceptions.InvalidIdTokenError):
    #         raise HTTPException(status_code=401, detail="Invalid ID token")
    #     elif isinstance(e, exceptions.EmailNotFoundError):
    #         raise HTTPException(status_code=401, detail="Email not found")
    #     elif isinstance(e, exceptions.WrongPasswordError):
    #         raise HTTPException(status_code=401, detail="Wrong password")
    #     else:
    #         raise HTTPException(status_code=401, detail="Authentication failed")

class NewsQuery(BaseModel):
    query: str

class Reference(BaseModel):
    title: str
    body: str

class SearchResponse(BaseModel):
    chatbot_response: str
    references: List[Reference]

@app.post("/search/news", response_model=SearchResponse)
def search_news(query: NewsQuery):
    # Simulate a chatbot response
    chatbot_response = f"Here are the latest news related to '{query.query}': [News 1], [News 2], [News 3]"

    # Simulate references
    references = [
        Reference(title="Reference 1", body="Reference 1 details..."),
        Reference(title="Reference 2", body="Reference 2 details..."),
        Reference(title="Reference 3", body="Reference 3 details..."),
    ]

    return SearchResponse(chatbot_response=chatbot_response, references=references)


class UserBehavior(BaseModel):
    token_id: str
    clicked_position: Optional[int] = None  # The position of the clicked news, or None if no click
    query: str  # The user's original query
    session_id: str


@app.post("/track/behavior")
async def track_user_behavior(behavior: UserBehavior):
    # You can implement your behavior tracking logic here
    if behavior.clicked_position is not None:
        # User clicked on a resource (news)
        clicked_position = behavior.clicked_position
        clicked_news = f"User clicked on news at position {clicked_position}"
        # You can log or process the click information here
    else:
        # User did not click on any resource
        clicked_news = "User did not click on any news"

    # You can use the user's original query as needed
    user_query = behavior.query

    # Return a response indicating the tracking result
    return {"message": "Behavior tracked successfully", "clicked_news": clicked_news}

all_interests = ["Interest 1", "Interest 2", "Interest 3"]

@app.get("/interests")
async def get_interests():
    return all_interests




class SelectedInterests(BaseModel):
    token_id: str
    interests: list
    session_id: str

@app.post("/select-interests")
async def select_interests(data: SelectedInterests):
    token_id = data.token_id
    interests = data.interests
    session_id = data.session_id

    # You can perform actions with the provided data (e.g., update user's interests)
    # Replace this with your specific logic

    return {"message": "Selected interests updated successfully"}



class PersonalSummary(BaseModel):
    token_id: str
    session_id: str
    summary: str

@app.post("/update-summary")
async def update_summary(data: PersonalSummary):
    token_id = data.token_id
    session_id = data.session_id
    summary = data.summary

    # You can perform actions with the provided data (e.g., update user's summary)
    # Replace this with your specific logic

    return {"message": "Personal summary updated successfully"}




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

