import { create } from "zustand";

import type {
  ChatMessage,
  Citation,
  UploadFileItem,
  WorkspaceDocument,
} from "@/lib/types";

interface WorkspaceState {
  workspaceId: string | null;
  documents: WorkspaceDocument[];
  uploadQueue: UploadFileItem[];
  activeDocumentId: string | null;
  highlightedBoundingBoxId: string | null;
  chatMessages: ChatMessage[];
  isChatLoading: boolean;
  viewerOpen: boolean;

  setWorkspaceId: (id: string) => void;
  setDocuments: (documents: WorkspaceDocument[]) => void;
  addUploadItems: (items: UploadFileItem[]) => void;
  updateUploadItem: (
    id: string,
    patch: Partial<UploadFileItem>,
  ) => void;
  promoteUploadToDocument: (
    uploadId: string,
    document: WorkspaceDocument,
  ) => void;
  updateDocument: (
    id: string,
    patch: Partial<WorkspaceDocument>,
  ) => void;
  setActiveDocument: (id: string | null) => void;
  toggleDocumentSelection: (id: string) => void;
  setHighlightedBoundingBox: (id: string | null) => void;
  setViewerOpen: (open: boolean) => void;
  addChatMessage: (message: ChatMessage) => void;
  setChatLoading: (loading: boolean) => void;
  navigateToCitation: (citation: Citation) => void;
}

export const useWorkspaceStore = create<WorkspaceState>((set, get) => ({
  workspaceId: null,
  documents: [],
  uploadQueue: [],
  activeDocumentId: null,
  highlightedBoundingBoxId: null,
  chatMessages: [],
  isChatLoading: false,
  viewerOpen: false,

  setWorkspaceId: (id) => set({ workspaceId: id }),

  setDocuments: (documents) => set({ documents }),

  addUploadItems: (items) =>
    set((state) => ({
      uploadQueue: [...state.uploadQueue, ...items],
    })),

  updateUploadItem: (id, patch) =>
    set((state) => ({
      uploadQueue: state.uploadQueue.map((item) =>
        item.id === id ? { ...item, ...patch } : item,
      ),
    })),

  promoteUploadToDocument: (uploadId, document) =>
    set((state) => ({
      uploadQueue: state.uploadQueue.filter((item) => item.id !== uploadId),
      documents: [...state.documents, document],
    })),

  updateDocument: (id, patch) =>
    set((state) => ({
      documents: state.documents.map((doc) =>
        doc.id === id ? { ...doc, ...patch } : doc,
      ),
    })),

  setActiveDocument: (id) =>
    set({ activeDocumentId: id, viewerOpen: id !== null }),

  toggleDocumentSelection: (id) =>
    set((state) => ({
      documents: state.documents.map((doc) =>
        doc.id === id ? { ...doc, selected: !doc.selected } : doc,
      ),
    })),

  setHighlightedBoundingBox: (id) =>
    set({ highlightedBoundingBoxId: id }),

  setViewerOpen: (open) => set({ viewerOpen: open }),

  addChatMessage: (message) =>
    set((state) => ({
      chatMessages: [...state.chatMessages, message],
    })),

  setChatLoading: (loading) => set({ isChatLoading: loading }),

  navigateToCitation: (citation) => {
    const { documents } = get();
    const doc = documents.find((d) => d.id === citation.documentId);
    if (!doc) return;

    set({
      activeDocumentId: citation.documentId,
      highlightedBoundingBoxId: citation.boundingBoxId,
      viewerOpen: true,
    });
  },
}));

export function getSelectedDocumentIds(documents: WorkspaceDocument[]): string[] {
  return documents.filter((doc) => doc.selected).map((doc) => doc.id);
}
