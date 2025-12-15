# Deployment Guide

This project consists of a Python Flask backend and a SolidJS frontend. You can deploy them together using Docker Compose or separately on cloud platforms.

## Prerequisites

- Docker and Docker Compose installed
- API Keys (e.g., `GEMINI_API_KEY4`)

## Option 1: Local / VPS Deployment (Docker Compose)

This is the easiest way to run the full stack.

1.  **Create a `.env` file** in the root directory with your API keys:
    ```bash
    GEMINI_API_KEY4=your_api_key_here
    ```

2.  **Build and Run**:
    ```bash
    docker-compose up --build -d
    ```

3.  **Access the Application**:
    - Frontend: `http://localhost:3000`
    - Backend: `http://localhost:5000`

## Option 2: Cloud Deployment (Render Blueprint)

The easiest way to deploy to Render is using the `render.yaml` Blueprint.

1.  **Push your code** to a GitHub/GitLab repository.
2.  **Go to Render Dashboard** and click "New" -> "Blueprint".
3.  **Connect your repository**.
4.  Render will automatically detect the `render.yaml` file and propose two services:
    *   `mindplex-backend` (Web Service)
    *   `mindplex-frontend` (Static Site)
5.  **Click "Apply"**.
6.  **Post-Creation Configuration**:
    *   **Backend**: Go to the `mindplex-backend` service -> **Environment** and add your `GEMINI_API_KEY4`.
    *   **Frontend**:
        1.  Wait for the `mindplex-backend` to finish deploying.
        2.  Copy the Backend URL (e.g., `https://mindplex-backend.onrender.com`).
        3.  Go to the `mindplex-frontend` service -> **Environment**.
        4.  Update `VITE_API_BASE_URL` with the Backend URL.
        5.  **Trigger a manual deploy** of the frontend (so it rebuilds with the correct API URL).

## Option 3: Manual Cloud Deployment (e.g., Railway, Heroku)

You can deploy the backend and frontend as separate services manually.

### Backend (Python)

1.  Create a new **Web Service** connected to this repository.
2.  **Root Directory**: `.` (Root)
3.  **Build Command**: `pip install -r experiments/requirements.txt`
4.  **Start Command**: `cd experiments && python mining_api.py`
    *   *Note: Ensure the platform supports Python 3.10+*
5.  **Environment Variables**:
    - Add `GEMINI_API_KEY4`
    - Add `PYTHON_VERSION` (if needed, e.g., `3.11.0`)

### Frontend (Static Site)

1.  Create a new **Static Site** connected to this repository.
2.  **Root Directory**: `experiments/atomspace_visualizer`
3.  **Build Command**: `npm install && npm run build`
4.  **Publish Directory**: `dist`
5.  **Environment Variables**:
    - `VITE_API_BASE_URL`: The URL of your deployed Backend (e.g., `https://my-backend.onrender.com`)

## Option 4: Manual Deployment

### Backend
1.  Navigate to `experiments/`.
2.  Install dependencies: `pip install -r requirements.txt`.
3.  Run: `python mining_api.py`.

### Frontend
1.  Navigate to `experiments/atomspace_visualizer/`.
2.  Install dependencies: `npm install`.
3.  Build: `npm run build`.
4.  Serve the `dist/` folder using any static file server (e.g., `serve -s dist`).
