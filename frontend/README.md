# Document Workspace Frontend

NotebookLM-style document workspace UI for the Sports Science RAG platform.

## Features

- **Drag-and-drop ingestion** — PDF-only uploads (max 50MB) with local UUID tracking and progressive status (`PENDING → UPLOADING → PROCESSING → COMPLETED/FAILED`)
- **Real-time parsing updates** — Server-Sent Events for granular sub-states (e.g. "Extracting tables", "Generating embeddings")
- **Split-pane layout** — Source manager (left), citation viewer (center), chat (right)
- **Interactive citations** — SVG bounding-box overlays on PDF pages; click chat citations to scroll and highlight sources
- **Multi-tenant security** — Axios interceptor attaches `Authorization: Bearer <JWT>` on every request; search queries include selected `document_ids` filters

## Quick start

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000/workspace/default](http://localhost:3000/workspace/default).

By default the app uses built-in mock API routes at `/api/v1` so you can demo uploads and chat without a backend.

## Environment

Copy `.env.example` to `.env.local`:

```bash
cp .env.example .env.local
```

| Variable | Description |
|----------|-------------|
| `NEXT_PUBLIC_API_URL` | Backend API base URL (default: `/api/v1` mock) |
| `NEXT_PUBLIC_DEMO_TENANT_ID` | Demo tenant ID embedded in dev JWT |

Point `NEXT_PUBLIC_API_URL` at your real backend (e.g. `http://localhost:8000/api/v1`) when integrating.

## Architecture

```
src/
├── app/                    # Next.js App Router pages & mock API routes
├── components/
│   ├── upload/             # DropZone, UploadProgressList
│   ├── workspace/          # SplitPaneLayout, CitationViewer, ChatPanel
│   └── ui/                 # shadcn-style primitives
├── hooks/                  # useUploadPipeline, useChat
├── lib/                    # api-client, auth, sse, types
└── stores/                 # Zustand workspace state
```

## Backend contract

All requests require `Authorization: Bearer <JWT>` with `tenant_id` in the payload.

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/workspaces/:id/documents/upload` | POST | Init upload, returns presigned URL |
| `/workspaces/:id/documents` | GET | List workspace documents |
| `/workspaces/:id/documents/:docId` | GET | Document layout + bounding boxes |
| `/workspaces/:id/documents/events` | GET (SSE) | Parsing progress stream |
| `/search/query` | POST | Chat query with `document_ids` filter |
