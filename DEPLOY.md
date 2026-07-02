# Deployment

## Backend: Hugging Face Spaces

Space URL:

- Repo page: https://huggingface.co/spaces/cjoyy/vacuum-rl
- Runtime URL: https://cjoyy-vacuum-rl.hf.space

Required files:

- `README.md` with Docker Space metadata:
  - `sdk: docker`
  - `app_port: 7860`
- `Dockerfile`
- `requirements.txt`
- `backend/`
- `ml/`

Deploy with the Hugging Face CLI:

```bash
hf auth login
hf upload cjoyy/vacuum-rl . . --repo-type=space \
  --exclude ".git/*" \
  --exclude ".venv/*" \
  --exclude "frontend/node_modules/*" \
  --exclude "frontend/dist/*" \
  --exclude "*.log"
```

After the Space finishes building, verify:

```bash
curl https://cjoyy-vacuum-rl.hf.space/health
curl https://cjoyy-vacuum-rl.hf.space/algorithms
```

Expected health response:

```json
{"status":"ok"}
```

## Frontend

The frontend is built inside the Docker image and served by FastAPI from the same Hugging Face Space origin.
No Vercel deployment or `VITE_API_URL` override is required.

## Production Smoke Test

1. Open the Vercel production URL.
2. Confirm the status pill says `connected`.
3. Click `Step`; the step counter should increase and the grid should update.
4. Click `Play`; the grid should continue updating through WebSocket messages.
5. Check browser console for failed WebSocket/CORS requests.
