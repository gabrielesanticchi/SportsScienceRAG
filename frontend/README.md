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

## Component Inventory

### Workspace Layout

| Component | File | Purpose |
|-----------|------|---------|
| **SplitPaneLayout** | `components/workspace/split-pane-layout.tsx` | Three-column layout container: left (SourceManager), center (CitationViewer), right (ChatPanel) |
| **SourceManager** | `components/workspace/source-manager.tsx` | Left panel: grid of document cards with checkboxes for selecting chat context |
| **CitationViewer** | `components/workspace/citation-viewer.tsx` | Center panel: react-pdf viewer with SVG bounding-box overlays; syncs with active document and highlighted citation |
| **ChatPanel** | `components/workspace/chat-panel.tsx` | Right panel: message history, chat input box, loading state; displays citations as clickable chips |

### Upload Flow

| Component | File | Purpose |
|-----------|------|---------|
| **DropZone** | `components/upload/drop-zone.tsx` | Drag-drop area for PDF files; validates file type and size (≤50MB) |
| **UploadProgressList** | `components/upload/upload-progress-list.tsx` | Displays upload queue with progress bars, status badges, and sub-state text |

### Hooks

| Hook | File | Purpose |
|------|------|---------|
| **useUploadPipeline** | `hooks/use-upload-pipeline.ts` | Orchestrates: drop → UUID → upload init → presigned URL → file PUT → SSE polling → store sync. Returns `uploadItems`, `dropZoneProps`, `handlers`. |
| **useChat** | `hooks/use-chat.ts` | Manages chat interaction: sends search query with selected document IDs, parses citations from response, updates store. Returns `sendMessage`, `isLoading`. |

### API & Utilities

| Module | File | Purpose |
|--------|------|---------|
| **api-client** | `lib/api-client.ts` | Axios instance with JWT interceptor; exports functions: `initDocumentUpload`, `uploadFileToStorage`, `listDocuments`, `getDocument`, `querySearch` |
| **auth** | `lib/auth.ts` | JWT token generation (dev only); includes `getAuthToken()`, `generateDevJWT(tenantId)` |
| **sse** | `lib/sse.ts` | Server-Sent Events helper; exports `streamParsingEvents(workspaceId)` which yields parsing update events |
| **types** | `lib/types.ts` | TypeScript interfaces: `WorkspaceDocument`, `ChatMessage`, `Citation`, `UploadFileItem`, `SearchQueryPayload`, etc. |

### State Management

| Store | File | Purpose |
|-------|------|---------|
| **workspace-store** | `stores/workspace-store.ts` | Zustand store managing: `documents[]`, `uploadQueue[]`, `chatMessages[]`, `activeDocumentId`, `highlightedBoundingBoxId`, selections, UI state |

## Data Flow Diagrams

### Upload Lifecycle

```
User drops PDF into DropZone
    ↓
useUploadPipeline validates (PDF, ≤50MB)
    ↓
Generate local UUID
    ↓
Call POST /workspaces/:id/documents/upload
    ← Response: { upload_token, presigned_url, document_id }
    ↓
Store: addUploadItems({ id: UUID, status: 'UPLOADING' })
    ↓
PUT file to presigned_url
    ├─ Track progress via onUploadProgress callback
    ├─ Store: updateUploadItem(UUID, { status: 'UPLOADING', progress: 0-100 })
    ↓
Call GET /workspaces/:id/documents/events (SSE)
    ├─ Poll EventSource for parsing updates
    ├─ On event: Store: updateUploadItem(UUID, { status: 'PROCESSING', substatus: '...' })
    ├─ Repeat until: { status: 'COMPLETED' } or { status: 'FAILED' }
    ↓
Store: promoteUploadToDocument(UUID, WorkspaceDocument)
    ├─ Remove from uploadQueue
    ├─ Add to documents[]
    ↓
SourceManager re-renders, upload item disappears
```

### Chat & Citation Cycle

```
User selects documents (checkboxes in SourceManager)
    ↓
User types query in ChatPanel input
    ↓
User presses Send
    ↓
useChat fires:
    POST /search/query {
        query: "...",
        document_ids: [selected UUIDs]
    }
    ↓
Backend returns: { answer: "...", citations: [...] }
    ↓
Store: addChatMessage({
    role: 'assistant',
    content: answer,
    citations: [...]
})
    ↓
ChatPanel renders assistant message with Citation chips
    ↓
User clicks Citation chip
    ↓
Store: navigateToCitation(citation)
    ├─ setActiveDocument(citation.document_id)
    ├─ CitationViewer scrolls to page
    ├─ SVG overlay highlights bounding_boxes
    ↓
Visual feedback: user sees source highlighted
```

## State Shape

The Zustand store tracks:

```typescript
{
  // Workspace context
  workspaceId: string | null

  // Document management
  documents: WorkspaceDocument[]           // Processed docs (id, filename, status, layout, bounding_boxes)
  uploadQueue: UploadFileItem[]            // In-flight uploads (id, filename, status, progress, substatus)

  // UI selection & viewing
  activeDocumentId: string | null          // Which document is open in CitationViewer
  highlightedBoundingBoxId: string | null  // Which bounding box to highlight (for citation focus)
  viewerOpen: boolean                      // Document modal open/closed

  // Chat
  chatMessages: ChatMessage[]              // Conversation (role: 'user'|'assistant', content, citations)
  isChatLoading: boolean                   // Search query in flight

  // Action creators (mutators)
  setWorkspaceId(id: string): void
  setDocuments(docs: WorkspaceDocument[]): void
  addUploadItems(items: UploadFileItem[]): void
  updateUploadItem(id: string, patch: Partial<UploadFileItem>): void
  promoteUploadToDocument(uploadId: string, doc: WorkspaceDocument): void
  updateDocument(id: string, patch: Partial<WorkspaceDocument>): void
  setActiveDocument(id: string | null): void
  toggleDocumentSelection(id: string): void
  setHighlightedBoundingBox(id: string | null): void
  setViewerOpen(open: boolean): void
  addChatMessage(message: ChatMessage): void
  setChatLoading(loading: boolean): void
  navigateToCitation(citation: Citation): void
}
```

## Backend Contract

All requests require `Authorization: Bearer <JWT>` with `tenant_id` in the payload.

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/workspaces/:id/documents/upload` | POST | Init upload, returns presigned URL |
| `/workspaces/:id/documents` | GET | List workspace documents |
| `/workspaces/:id/documents/:docId` | GET | Document layout + bounding boxes |
| `/workspaces/:id/documents/events` | GET (SSE) | Parsing progress stream |
| `/search/query` | POST | Chat query with `document_ids` filter |

For detailed request/response schemas, see [`docs/CODEMAPS/frontend.md`](../CODEMAPS/frontend.md#backend-contract).
