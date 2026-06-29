import type {
  BoundingBox,
  ParsingEvent,
  UploadStatus,
  WorkspaceDocument,
} from "@/lib/types";

interface MockDocumentRecord extends WorkspaceDocument {
  fileBuffer?: ArrayBuffer;
}

const documentsByWorkspace = new Map<string, Map<string, MockDocumentRecord>>();
const parsingListeners = new Map<string, Set<(event: ParsingEvent) => void>>();

const PARSING_STAGES = [
  "Extracting text",
  "Detecting sections",
  "Extracting tables",
  "Generating embeddings",
  "Indexing vectors",
];

function getWorkspaceStore(workspaceId: string): Map<string, MockDocumentRecord> {
  let store = documentsByWorkspace.get(workspaceId);
  if (!store) {
    store = new Map();
    documentsByWorkspace.set(workspaceId, store);
  }
  return store;
}

function emitParsingEvent(workspaceId: string, event: ParsingEvent): void {
  const listeners = parsingListeners.get(workspaceId);
  listeners?.forEach((listener) => listener(event));
}

function generateMockBoundingBoxes(pageCount: number): BoundingBox[] {
  const boxes: BoundingBox[] = [];
  for (let page = 1; page <= pageCount; page += 1) {
    boxes.push(
      {
        id: `bb-${page}-header`,
        page,
        x: 48,
        y: 48,
        width: 500,
        height: 36,
        type: "header",
        text: `Section heading (page ${page})`,
      },
      {
        id: `bb-${page}-para`,
        page,
        x: 48,
        y: 100,
        width: 500,
        height: 120,
        type: "paragraph",
        text: `Body paragraph content on page ${page}.`,
      },
      {
        id: `bb-${page}-table`,
        page,
        x: 48,
        y: 240,
        width: 500,
        height: 80,
        type: "table",
        text: `Table region on page ${page}`,
      },
    );
  }
  return boxes;
}

export function listDocuments(workspaceId: string): WorkspaceDocument[] {
  const store = getWorkspaceStore(workspaceId);
  return Array.from(store.values()).map(({ fileBuffer: _, ...doc }) => doc);
}

export function getDocument(
  workspaceId: string,
  documentId: string,
): WorkspaceDocument | null {
  const doc = getWorkspaceStore(workspaceId).get(documentId);
  if (!doc) return null;
  const { fileBuffer: _, ...rest } = doc;
  return rest;
}

export function createDocument(
  workspaceId: string,
  filename: string,
): { documentId: string; uploadToken: string } {
  const documentId = crypto.randomUUID();
  const uploadToken = crypto.randomUUID();

  const doc: MockDocumentRecord = {
    id: documentId,
    name: filename,
    status: "UPLOADING",
    boundingBoxes: [],
    selected: true,
    uploadedAt: new Date().toISOString(),
    pageCount: 3,
  };

  getWorkspaceStore(workspaceId).set(documentId, doc);
  pendingUploads.set(uploadToken, { workspaceId, documentId });

  return { documentId, uploadToken };
}

const pendingUploads = new Map<
  string,
  { workspaceId: string; documentId: string }
>();

export function completeUpload(uploadToken: string, buffer: ArrayBuffer): boolean {
  const pending = pendingUploads.get(uploadToken);
  if (!pending) return false;

  const doc = getWorkspaceStore(pending.workspaceId).get(pending.documentId);
  if (!doc) return false;

  doc.fileBuffer = buffer;
  doc.pdfUrl = `/api/v1/documents/${pending.documentId}/pdf?token=${uploadToken}`;
  doc.status = "PROCESSING";
  pendingUploads.delete(uploadToken);

  void simulateParsing(pending.workspaceId, pending.documentId);
  return true;
}

async function simulateParsing(
  workspaceId: string,
  documentId: string,
): Promise<void> {
  const store = getWorkspaceStore(workspaceId);
  const doc = store.get(documentId);
  if (!doc) return;

  for (let index = 0; index < PARSING_STAGES.length; index += 1) {
    await new Promise((resolve) => setTimeout(resolve, 800));
    const progress = Math.round(((index + 1) / PARSING_STAGES.length) * 100);
    doc.status = "PROCESSING";

    emitParsingEvent(workspaceId, {
      documentId,
      status: "PROCESSING",
      progress,
      subState: PARSING_STAGES[index],
    });
  }

  doc.status = "COMPLETED";
  doc.boundingBoxes = generateMockBoundingBoxes(doc.pageCount ?? 3);

  emitParsingEvent(workspaceId, {
    documentId,
    status: "COMPLETED",
    progress: 100,
    subState: "Ready",
  });
}

export function subscribeParsingEvents(
  workspaceId: string,
  listener: (event: ParsingEvent) => void,
): () => void {
  let listeners = parsingListeners.get(workspaceId);
  if (!listeners) {
    listeners = new Set();
    parsingListeners.set(workspaceId, listeners);
  }
  listeners.add(listener);
  return () => listeners?.delete(listener);
}

export function getDocumentPdf(documentId: string): ArrayBuffer | null {
  for (const store of documentsByWorkspace.values()) {
    const doc = store.get(documentId);
    if (doc?.fileBuffer) return doc.fileBuffer;
  }
  return null;
}

export function updateDocumentStatus(
  workspaceId: string,
  documentId: string,
  status: UploadStatus,
): void {
  const doc = getWorkspaceStore(workspaceId).get(documentId);
  if (doc) doc.status = status;
}
