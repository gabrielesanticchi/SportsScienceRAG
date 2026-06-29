"use client";

import { useCallback, useEffect } from "react";
import { v4 as uuidv4 } from "uuid";

import {
  fetchDocumentLayout,
  getParsingEventsUrl,
  initDocumentUpload,
  uploadFileToStorage,
} from "@/lib/api-client";
import { subscribeToParsingEvents } from "@/lib/sse";
import {
  ACCEPTED_MIME_TYPES,
  MAX_PDF_SIZE_BYTES,
  type UploadFileItem,
  type UploadStatus,
  type WorkspaceDocument,
} from "@/lib/types";
import { useWorkspaceStore } from "@/stores/workspace-store";

function validatePdfFile(file: File): string | null {
  if (!ACCEPTED_MIME_TYPES.includes(file.type) && !file.name.toLowerCase().endsWith(".pdf")) {
    return "Only PDF files are accepted.";
  }
  if (file.size > MAX_PDF_SIZE_BYTES) {
    return "File exceeds the 50MB size limit.";
  }
  return null;
}

export function useUploadPipeline(workspaceId: string) {
  const addUploadItems = useWorkspaceStore((s) => s.addUploadItems);
  const updateUploadItem = useWorkspaceStore((s) => s.updateUploadItem);
  const promoteUploadToDocument = useWorkspaceStore((s) => s.promoteUploadToDocument);
  const updateDocument = useWorkspaceStore((s) => s.updateDocument);

  useEffect(() => {
    const unsubscribe = subscribeToParsingEvents(
      getParsingEventsUrl(workspaceId),
      (event) => {
        updateDocument(event.documentId, {
          status: event.status,
          subState: event.subState,
          progress: event.progress,
        });

        if (event.status === "COMPLETED") {
          void fetchDocumentLayout(workspaceId, event.documentId)
            .then((doc) => updateDocument(event.documentId, doc))
            .catch(() => undefined);
        }
      },
    );
    return unsubscribe;
  }, [workspaceId, updateDocument]);

  const processUpload = useCallback(
    async (item: UploadFileItem) => {
      try {
        updateUploadItem(item.id, { status: "UPLOADING", progress: 0 });

        const { document_id, upload_url } = await initDocumentUpload(
          workspaceId,
          item.name,
          item.size,
        );

        updateUploadItem(item.id, { documentId: document_id, progress: 5 });

        await uploadFileToStorage(upload_url, item.file, (percent) => {
          updateUploadItem(item.id, { progress: Math.max(5, percent) });
        });

        updateUploadItem(item.id, {
          status: "PROCESSING",
          progress: 100,
          subState: "Queued for parsing",
        });

        const document: WorkspaceDocument = {
          id: document_id,
          name: item.name,
          status: "PROCESSING",
          boundingBoxes: [],
          selected: true,
          uploadedAt: new Date().toISOString(),
          pdfUrl: URL.createObjectURL(item.file),
        };

        promoteUploadToDocument(item.id, document);
      } catch (error) {
        const message =
          error instanceof Error ? error.message : "Upload failed";
        updateUploadItem(item.id, {
          status: "FAILED",
          error: message,
        });
      }
    },
    [workspaceId, updateUploadItem, promoteUploadToDocument],
  );

  const enqueueFiles = useCallback(
    (files: FileList | File[]) => {
      const fileArray = Array.from(files);
      const items: UploadFileItem[] = [];

      for (const file of fileArray) {
        const validationError = validatePdfFile(file);
        const item: UploadFileItem = {
          id: uuidv4(),
          file,
          name: file.name,
          size: file.size,
          status: validationError ? "FAILED" : "PENDING",
          progress: 0,
          error: validationError ?? undefined,
        };
        items.push(item);
      }

      addUploadItems(items);

      for (const item of items) {
        if (item.status === "PENDING") {
          void processUpload(item);
        }
      }
    },
    [addUploadItems, processUpload],
  );

  return { enqueueFiles };
}

export function statusLabel(status: UploadStatus, subState?: string): string {
  if (subState) return subState;
  switch (status) {
    case "PENDING":
      return "Validating…";
    case "UPLOADING":
      return "Uploading to storage…";
    case "PROCESSING":
      return "Processing document…";
    case "COMPLETED":
      return "Ready";
    case "FAILED":
      return "Failed";
    default: {
      const _exhaustive: never = status;
      return _exhaustive;
    }
  }
}
