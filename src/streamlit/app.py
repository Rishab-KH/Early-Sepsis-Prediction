import os
import numpy as np
import pandas as pd
import streamlit as st
import requests
from dotenv import load_dotenv

load_dotenv()

streamlit_uri = os.getenv('streamlit_uri')