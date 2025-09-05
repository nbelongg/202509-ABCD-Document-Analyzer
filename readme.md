# ABCD_chatbot

## Description

ABCD chatbot consists of three main components: Admin APIs, FastAPI backend, and Streamlit frontend.

## Table of Contents

- [Components](#components)
- [Installation](#installation)
- [Usage](#usage)

## Components

1. **Admin APIs**: Located in the `admin_apis` directory, these APIs handle fetching, updating, and deleting analyzer/evaluator prompts.

2. **FastAPI Backend**: The main backend service (`abcd_fastapi_main.py`) that contains the code for analyzer and evaluator APIs.

3. **Streamlit Frontend**: A UI version of the APIs (`abcd_streamlit_main.py`) deployed on a separate port.

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/PradipNichite/abcd_chatbot.git
   ```

2. Navigate to the project directory:

   ```
   cd abcd_chatbot
   ```

3. Checkout the new-analyzer branch:

   ```
   git checkout new-analyzer
   ```

4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
