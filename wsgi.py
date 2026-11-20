"""Production entry point:  gunicorn wsgi:app"""
from memora import create_app

app = create_app()
