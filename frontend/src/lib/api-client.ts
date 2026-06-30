import axios, { type AxiosInstance } from "axios";

import { getAuthToken } from "@/lib/auth";
import type {
  Citation,
  SearchQueryPayload,
  UploadInitResponse,
  WorkspaceDocument,
} from "@/lib/types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "/api/v1";

export const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

apiClient.interceptors.request.use((config) => {
  const token = getAuthToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export async function initDocumentUpload(
  workspaceId: string,
  filename: string,
  sizeBytes: number,
): Promise<UploadInitResponse> {
  const { data } = await apiClient.post<UploadInitResponse>(
    `/workspaces/${workspaceId}/documents/upload`,
    { filename, size_bytes: sizeBytes },
  );
  return data;
}

export async function uploadFileToStorage(
  uploadUrl: string,
  file: File,
  onProgress?: (percent: number) => void,
): Promise<void> {
  await axios.put(uploadUrl, file, {
    headers: { "Content-Type": "application/pdf" },
    onUploadProgress: (event) => {
      if (!event.total || !onProgress) return;
      onProgress(Math.round((event.loaded / event.total) * 100));
    },
  });
}

export async function fetchWorkspaceDocuments(
  workspaceId: string,
): Promise<WorkspaceDocument[]> {
  const { data } = await apiClient.get<{ documents: WorkspaceDocument[] }>(
    `/workspaces/${workspaceId}/documents`,
  );
  return data.documents;
}

export async function fetchDocumentLayout(
  workspaceId: string,
  documentId: string,
): Promise<WorkspaceDocument> {
  const { data } = await apiClient.get<WorkspaceDocument>(
    `/workspaces/${workspaceId}/documents/${documentId}`,
  );
  return data;
}

export async function submitSearchQuery(payload: SearchQueryPayload): Promise<{
  answer: string;
  citations: Citation[];
}> {
  const { data } = await apiClient.post<{
    answer: string;
    citations: Citation[];
  }>("/search/query", payload);
  return data;
}

export function getParsingEventsUrl(workspaceId: string): string {
  const base = API_BASE_URL.startsWith("http")
    ? API_BASE_URL
    : `${typeof window !== "undefined" ? window.location.origin : ""}${API_BASE_URL}`;
  return `${base}/workspaces/${workspaceId}/documents/events`;
}
