# Frontend Codemap

**Last Updated:** 2026-06-30  
**Framework:** Next.js 16 (App Router), React 19, TypeScript  
**Entry Points:** `frontend/src/app/workspace/[workspaceId]/page.tsx`

## Overview

The Document Workspace Frontend is a NotebookLM-style collaborative document interface built with Next.js 16. It provides drag-and-drop PDF ingestion, real-time parsing status via Server-Sent Events, and a split-pane workspace for managing sources, viewing citations, and conducting multi-document chat.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│             Next.js Frontend (port 3000)            │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌─────────────────────────────────────────────┐    │
│  │  App Router Pages & Mock API Routes          │    │
│  │  ├─ /workspace/[workspaceId]                 │    │
│  │  ├─ /api/v1/workspaces/*/documents/*         │    │
│  │  ├─ /api/v1/search/query                     │    │
│  │  └─ /api/v1/upload/*                         │    │
│  └─────────────────────────────────────────────┘    │
│           ↓                     ↓                     │
│  ┌─────────────────────────────────────────────┐    │
│  │  React Components (SplitPaneLayout)         │    │
│  │  ├─ SourceManager (grid view, checkboxes)   │    │
│  │  ├─ CitationViewer (react-pdf + SVG)        │    │
│  │  └─ ChatPanel (messages + chat input)       │    │
│  └─────────────────────────────────────────────┘    │
│           ↓                     ↓                     │
│  ┌─────────────────────────────────────────────┐    │
│  │  Zustand Store (workspace-store.ts)         │    │
│  │  ├─ documents[], uploadQueue[], messages[]  │    │
│  │  ├─ activeDocumentId, highlightedBBox      │    │
│  │  └─ Selection, citation navigation          │    │
│  └─────────────────────────────────────────────┘    │
│           ↓                     ↓                     │
│  ┌─────────────────────────────────────────────┐    │
│  │  Hooks & API Client                         │    │
│  │  ├─ useUploadPipeline (UUID, SSE polling)   │    │
│  │  ├─ useChat (search.query + citations)      │    │
│  │  ├─ api-client.ts (Axios + JWT interceptor) │    │
│  │  └─ sse.ts (EventSource helpers)            │    │
│  └─────────────────────────────────────────────┘    │
│           ↓                                          │
│  ┌─────────────────────────────────────────────┐    │
│  │  Python Backend (or Mock at /api/v1)        │    │
│  │  REST + SSE + JWT Authentication            │    │
│  └─────────────────────────────────────────────┘    │
│                                                      │
└─────────────────────────────────────────────────────┘
```

## Directory Structure

```
frontend/
├── src/
│   ├── app/                            # Next.js App Router
│   │   ├── layout.tsx                  # Root layout, providers setup
│   │   ├── page.tsx                    # Home page
│   │   ├── workspace/
│   │   │   └── [workspaceId]/
│   │   │       └── page.tsx            # Main workspace component
│   │   └── api/v1/                     # Mock backend routes
│   │       ├── workspaces/[workspaceId]/documents/
│   │       │   ├── upload/route.ts     # Init upload, return presigned URL
│   │       │   ├── route.ts            # List documents
│   │       │   ├── [documentId]/route.ts  # Get document layout
│   │       │   └── events/route.ts     # SSE: parsing progress stream
│   │       ├── search/query/route.ts   # Mock search with document filtering
│   │       ├── upload/[uploadToken]/route.ts  # Finalize upload
│   │       └── documents/[documentId]/pdf/route.ts  # Serve PDF
│   │
│   ├── components/
│   │   ├── providers/
│   │   │   ├── app-providers.tsx       # Root provider wrapper (Zustand + React context)
│   │   │   └── auth-provider.tsx       # JWT token management
│   │   │
│   │   ├── workspace/                  # Main UI components
│   │   │   ├── split-pane-layout.tsx   # Three-column layout shell
│   │   │   ├── source-manager.tsx      # Left panel: document grid
│   │   │   ├── citation-viewer.tsx     # Center: PDF viewer + SVG overlays
│   │   │   ├── chat-panel.tsx          # Right: messages + input
│   │   │   └── document-viewer-modal.tsx  # Full-screen document view
│   │   │
│   │   ├── upload/
│   │   │   ├── drop-zone.tsx           # Drag-drop area
│   │   │   └── upload-progress-list.tsx  # Upload queue + progress bars
│   │   │
│   │   └── ui/                         # shadcn-style primitives
│   │       ├── button.tsx, badge.tsx, checkbox.tsx, etc.
│   │       └── scroll-area.tsx, dialog.tsx, progress.tsx
│   │
│   ├── hooks/
│   │   ├── use-upload-pipeline.ts      # Orchestrates: drop → UUID → upload → SSE polling → store update
│   │   └── use-chat.ts                 # Search.query + adds citations to store
│   │
│   ├── lib/
│   │   ├── api-client.ts               # Axios instance + JWT interceptor
│   │   ├── auth.ts                     # Dev JWT token generation
│   │   ├── sse.ts                      # Server-Sent Events helper
│   │   └── types.ts                    # TypeScript interfaces: WorkspaceDocument, ChatMessage, Citation, etc.
│   │
│   └── stores/
│       └── workspace-store.ts          # Zustand store: documents, messages, selections, UI state
│
├── public/                             # Static assets
├── .env.example                        # Environment variables template
├── package.json                        # Dependencies: Next.js 16, React 19, react-pdf, Zustand, Axios
├── next.config.js                      # Next.js configuration
├── tsconfig.json                       # TypeScript configuration
└── README.md                           # Frontend documentation

```

## Key Modules

| Module | Purpose | Key Exports | Dependencies |
|--------|---------|-----------|--------------|
| **workspace-store.ts** | Zustand state container for entire workspace | `useWorkspaceStore` | zustand |
| **api-client.ts** | Axios HTTP client with JWT authorization | `apiClient`, `initDocumentUpload`, `listDocuments`, `querySearch` | axios |
| **use-upload-pipeline.ts** | Orchestrates PDF upload: drop → UUID → upload → SSE polling → store sync | `useUploadPipeline` | Zustand, api-client, sse |
| **use-chat.ts** | Sends search query to backend, parses citations, updates store | `useChat` | Zustand, api-client |
| **sse.ts** | Manages EventSource connections, parsing status updates | `streamParsingEvents` | EventSource |
| **types.ts** | TypeScript interfaces for API contracts | `WorkspaceDocument`, `ChatMessage`, `Citation`, `SearchQueryPayload` | — |
| **SplitPaneLayout** | Three-column container layout | React component | Radix UI |
| **SourceManager** | Left panel: document grid, checkboxes for chat context | React component | Zustand, shadcn UI |
| **CitationViewer** | Center: react-pdf viewer with SVG bounding box overlays | React component | react-pdf |
| **ChatPanel** | Right panel: message list, chat input, citation chips | React component | Zustand, shadcn UI |
| **DropZone** | Drag-drop area, file validation (PDF, ≤50MB) | React component | shadcn UI |
| **UploadProgressList** | Upload queue progress bars, status badges | React component | Zustand, shadcn UI |

## Data Flow

### 1. Document Upload Pipeline

```
User drags PDF
    ↓
DropZone validates (PDF, ≤50MB)
    ↓
useUploadPipeline: generate local UUID
    ↓
Call POST /workspaces/:id/documents/upload
    ├─ Backend returns { upload_token, presigned_url, document_id }
    ├─ Store: addUploadItems({ id: UUID, status: 'UPLOADING' })
    ↓
PUT file to presigned_url (axios with onProgress)
    ├─ Track upload % in store via updateUploadItem()
    ↓
Call GET /workspaces/:id/documents/events (SSE)
    ├─ Stream: { document_id, sub_state: 'Extracting tables', progress: 42 }
    ├─ Store: updateUploadItem(UUID, { status: 'PROCESSING', substatus: 'Extracting tables' })
    ├─ Repeat until: { status: 'COMPLETED' } or { status: 'FAILED' }
    ↓
promoteUploadToDocument(UUID, WorkspaceDocument)
    ├─ Remove from uploadQueue
    ├─ Add to documents[] with full metadata (layout, bounding_boxes)
    ↓
SourceManager re-renders, UploadProgressList disappears
```

### 2. Chat & Citation Flow

```
User selects documents (checkboxes in SourceManager)
    ↓
User types query in ChatPanel input
    ↓
useChat hook fires:
    POST /search/query {
        query: "...",
        document_ids: [selected UUIDs]
    }
    ↓
Backend returns:
    {
        answer: "...",
        citations: [
            {
                document_id: "...",
                page: 3,
                bounding_boxes: [{x, y, w, h, text: "..."}]
            }
        ]
    }
    ↓
Store: addChatMessage({
        role: 'assistant',
        content: answer,
        citations: [...]
    })
    ↓
ChatPanel renders message with Citation chips
    ↓
User clicks Citation chip
    ↓
navigateToCitation(citation):
    ├─ setActiveDocument(citation.document_id)
    ├─ CitationViewer scrolls to page
    ├─ SVG overlay highlights bounding_boxes
    ├─ setHighlightedBoundingBox(id) for visual feedback
```

### 3. State Management (Zustand)

```
WorkspaceState {
    // Workspace context
    workspaceId: string | null
    
    // Documents & uploads
    documents: WorkspaceDocument[]          // Processed docs with layout
    uploadQueue: UploadFileItem[]           // In-flight uploads (UUID, status, progress)
    
    // UI selection & viewing
    activeDocumentId: string | null         // Which doc is open in CitationViewer
    highlightedBoundingBoxId: string | null // Which bbox to highlight
    viewerOpen: boolean                      // Document modal visibility
    
    // Chat
    chatMessages: ChatMessage[]             // Conversation history
    isChatLoading: boolean                  // Search in flight
    
    // Actions: setters, updaters, navigation
    setWorkspaceId(id)
    addUploadItems(items)
    updateUploadItem(id, patch)
    promoteUploadToDocument(uploadId, doc)
    updateDocument(id, patch)
    toggleDocumentSelection(id)
    navigateToCitation(citation)
    addChatMessage(msg)
    ...
}
```

## Backend Contract

### Authentication

All requests require the HTTP header:
```
Authorization: Bearer <JWT>
```

The JWT payload must include `tenant_id` (string). In development, JWT tokens are generated by `lib/auth.ts` with a demo tenant.

### Endpoints

#### 1. Upload Initiation

**POST** `/workspaces/:workspaceId/documents/upload`

Request:
```json
{
  "filename": "research_paper.pdf",
  "size_bytes": 5242880
}
```

Response (`200 OK`):
```json
{
  "document_id": "uuid-v4",
  "upload_token": "token-string",
  "presigned_url": "https://storage.example.com/upload/token?..."
}
```

The frontend then PUTs the file to `presigned_url` with `Content-Type: application/pdf`.

#### 2. List Documents

**GET** `/workspaces/:workspaceId/documents`

Response (`200 OK`):
```json
{
  "documents": [
    {
      "id": "uuid-v4",
      "filename": "research_paper.pdf",
      "status": "COMPLETED",
      "num_pages": 25,
      "layout": [
        {
          "page": 1,
          "width": 612.0,
          "height": 792.0
        }
      ],
      "created_at": "2026-06-30T12:00:00Z",
      "updated_at": "2026-06-30T12:05:00Z"
    }
  ]
}
```

#### 3. Get Document Layout & Bounding Boxes

**GET** `/workspaces/:workspaceId/documents/:documentId`

Response (`200 OK`):
```json
{
  "id": "uuid-v4",
  "filename": "research_paper.pdf",
  "status": "COMPLETED",
  "num_pages": 25,
  "layout": [
    {
      "page": 1,
      "width": 612.0,
      "height": 792.0,
      "bounding_boxes": [
        {
          "id": "bbox-1",
          "page": 1,
          "x": 50.0,
          "y": 100.0,
          "width": 500.0,
          "height": 50.0,
          "text": "Introduction"
        }
      ]
    }
  ]
}
```

#### 4. Parsing Progress Stream (SSE)

**GET** `/workspaces/:workspaceId/documents/events` (Server-Sent Events)

The frontend opens an EventSource connection. The backend streams events as parsing progresses:

```
event: parsing_update
data: {"document_id": "uuid-v4", "sub_state": "Extracting text", "progress": 25}

event: parsing_update
data: {"document_id": "uuid-v4", "sub_state": "Generating embeddings", "progress": 75}

event: parsing_complete
data: {"document_id": "uuid-v4", "status": "COMPLETED"}
```

Or on error:
```
event: parsing_error
data: {"document_id": "uuid-v4", "status": "FAILED", "error_message": "Invalid PDF structure"}
```

#### 5. Search Query

**POST** `/search/query`

Request:
```json
{
  "query": "What is high intensity interval training?",
  "document_ids": ["uuid-1", "uuid-2"],
  "limit": 10
}
```

Response (`200 OK`):
```json
{
  "answer": "High intensity interval training (HIIT) is...",
  "citations": [
    {
      "document_id": "uuid-1",
      "page": 3,
      "bounding_boxes": [
        {
          "id": "bbox-42",
          "x": 50.0,
          "y": 150.0,
          "width": 500.0,
          "height": 60.0,
          "text": "HIIT improves cardiovascular fitness..."
        }
      ]
    }
  ]
}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | `/api/v1` | Backend API base URL. Set to `http://localhost:8000/api/v1` for local Python backend. |
| `NEXT_PUBLIC_DEMO_TENANT_ID` | `demo-tenant` | Tenant ID embedded in dev JWT. Used in mock API routes. |

Copy `.env.example` to `.env.local` to override defaults:

```bash
cp frontend/.env.example frontend/.env.local
# Edit .env.local as needed
```

## Integration with Python Backend

When integrating with the Python RAG backend:

1. **Ensure JWT authentication** — The backend must decode the `Authorization: Bearer <JWT>` header and extract `tenant_id` from the token payload.

2. **Implement SSE endpoint** — The `/documents/events` endpoint must stream parsing status as Server-Sent Events. The frontend polls this indefinitely until a `parsing_complete` or `parsing_error` event arrives.

3. **Support document filtering** — The `/search/query` endpoint must accept a `document_ids[]` array and filter results accordingly (or return results from all documents if the array is empty).

4. **Return valid schemas** — All responses must match the JSON schemas described above. The frontend will break if required fields are missing.

5. **Set CORS headers** — If running frontend and backend on different origins, the backend must include appropriate CORS headers (typically already handled by FastAPI/Flask middleware).

## Dependencies

### Production

| Package | Version | Purpose |
|---------|---------|---------|
| `next` | 16.2.9 | App Router, SSR, API routes |
| `react` | 19.2.4 | UI rendering |
| `react-dom` | 19.2.4 | DOM rendering |
| `react-pdf` | 10.4.1 | PDF viewer |
| `axios` | 1.18.1 | HTTP client |
| `zustand` | 5.0.14 | State management |
| `uuid` | 14.0.1 | Local UUID generation |
| `@radix-ui/*` | ^1.x | UI primitives (checkbox, dialog, progress, scroll-area, etc.) |
| `lucide-react` | 1.22.0 | Icons |
| `tailwind-merge`, `class-variance-authority`, `clsx` | Latest | Styling utilities |

### Development

- ESLint
- TypeScript

## Quick Start for Developers

1. **Install dependencies**
   ```bash
   cd frontend
   npm install
   ```

2. **Set up environment (optional)**
   ```bash
   cp .env.example .env.local
   # Edit .env.local to point to your backend, or keep defaults for mock API
   ```

3. **Run dev server**
   ```bash
   npm run dev
   ```
   The frontend will be available at `http://localhost:3000/workspace/default`.

4. **Try the features**
   - Upload a PDF using the drop zone (left panel)
   - Wait for parsing to complete (progress updates via SSE)
   - Select documents with checkboxes
   - Type a query and send (mock backend will respond)
   - Click citations to navigate the source document

5. **Build for production**
   ```bash
   npm run build
   npm start
   ```

## Testing & Debugging

- **Browser DevTools** — Inspect Network tab to see API requests (including JWT header), Console for errors, Application tab for localStorage (Zustand hydration).
- **Mock API routes** — Located at `frontend/src/app/api/v1/`. Modify these routes to simulate backend behavior for development.
- **SSE debugging** — Open DevTools > Network, find the `/documents/events` request, inspect the streaming events in the Details tab.

## Related Codemaps

- **Backend** — Python RAG engine at `sports_science_search/` (see root [README.md](../../README.md#architecture))
- **Integration notes** — See "Backend integration" section above for API contract details
