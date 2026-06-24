// web/config.js
//
// The ONE line you edit before deploying to Vercel: point this at your
// deployed backend (Render/Fly/HF Spaces -- NOT Vercel, see README.md
// for why the FastAPI+ML backend can't live on Vercel itself).
//
// Local dev: leave as-is (assumes `uvicorn backend.main:app --port 8000`
// running on the same machine).
const BACKEND_URL = "http://localhost:8000";
