from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session

# Define the database URL
DATABASE_URL = "sqlite:///./test.db"

# Create an SQLAlchemy engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# ... (your existing imports)

# Database Models
class DB_Interest(Base):
    __tablename__ = "interests"
    interest_id = Column(Integer, primary_key=True, index=True)
    interest_name = Column(String, index=True)

class DB_User(Base):
    __tablename__ = "users"
    user_id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String)
    last_name = Column(String)
    email = Column(String)
    firebase_id = Column(String, index=True)

class DB_UserInterest(Base):
    __tablename__ = "user_interests"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    interest_id = Column(Integer, ForeignKey("interests.interest_id"))


    user = relationship("DB_User", back_populates="user_interests")
    interest = relationship("DB_Interest", back_populates="user_interests")

DB_User.user_interests = relationship("DB_UserInterest", back_populates="user")
DB_Interest.user_interests = relationship("DB_UserInterest", back_populates="interest")

# Drop existing tables if they exist
Base.metadata.drop_all(bind=engine)

Base.metadata.create_all(bind=engine)

# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# # Example: Inserting Sample Data
# def create_sample_data(db: Session):
#     # Create sample interests
#     interest_tech = Interest(interest_name="Technology")
#     interest_sport = Interest(interest_name="Sports")
#     db.add_all([interest_tech, interest_sport])
#     db.commit()

#     # Create a sample user
#     user = User(firebase_id = '4x89mfstotM4yxmcG6Xb552EmGJ3', first_name="Qazi" last_name="Saad", email="qazisaad@live.com")
#     db.add(user)
#     db.commit()

#     # Associate interests with the user
#     user_interest_tech = UserInterest(user_id=user.user_id, interest_id=interest_tech.interest_id)
#     user_interest_sport = UserInterest(user_id=user.user_id, interest_id=interest_sport.interest_id)
#     db.add_all([user_interest_tech, user_interest_sport])
#     db.commit()

# # Example: Querying Data
# def query_sample_data(db: Session):
#     # Query interests
#     interests = db.query(Interest).all()
#     print("Interests:")
#     for interest in interests:
#         print(f"Interest ID: {interest.interest_id}, Name: {interest.interest_name}")

#     # Query users
#     users = db.query(User).all()
#     print("\nUsers:")
#     for user in users:
#         print(f"User ID: {user.user_id}, First Name: {user.first_name}, Last Name: {user.last_name}, Email: {user.email}, Firebase ID: {user.firebase_id}")

#     # Query user interests with associated user and interest data
#     user_interests = db.query(UserInterest).\
#         join(UserInterest.user).\
#         join(UserInterest.interest).all()
#     print("\nUser Interests:")
#     for user_interest in user_interests:
#         print(f"User ID: {user_interest.user.user_id}, Name: {user_interest.user.name}, "
#               f"Interest: {user_interest.interest.interest_name}")
        

# with SessionLocal() as db:
#     create_sample_data(db)
#     query_sample_data(db)