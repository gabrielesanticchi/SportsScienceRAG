export type UploadStatus =
  | "PENDING"
  | "UPLOADING"
  | "PROCESSING"
  | "COMPLETED"
  | "FAILED";

export type BoundingBoxType = "paragraph" | "header" | "table";

export interface BoundingBox {
  id: string;
  page: number;
  x: number;
  y: number;
  width: number;
  height: number;
  type: BoundingBoxType;
  text?: string;
  estimated?: boolean;
}

export interface UploadFileItem {
  id: string;
  file: File;
  name: string;
  size: number;
  status: UploadStatus;
  progress: number;
  subState?: string;
  error?: string;
  documentId?: string;
}

export interface WorkspaceDocument {
  id: string;
  name: string;
  status: UploadStatus;
  pageCount?: number;
  boundingBoxes: BoundingBox[];
  pdfUrl?: string;
  selected: boolean;
  uploadedAt: string;
  subState?: string;
  progress?: number;
}

export interface Citation {
  id: string;
  documentId: string;
  boundingBoxId: string;
  text: string;
  page: number;
  label: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  timestamp: string;
}

export interface SearchQueryPayload {
  query: string;
  workspace_id: string;
  document_ids: string[];
}

export interface ParsingEvent {
  documentId: string;
  status: UploadStatus;
  progress: number;
  subState?: string;
  error?: string;
}

export interface UploadInitResponse {
  document_id: string;
  upload_url: string;
}

export const MAX_PDF_SIZE_BYTES = 50 * 1024 * 1024;
export const ACCEPTED_MIME_TYPES = ["application/pdf"];
