"use client";

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { CitationViewer } from "@/components/workspace/citation-viewer";
import type { Citation, WorkspaceDocument } from "@/lib/types";

interface DocumentViewerModalProps {
  open: boolean;
  document: WorkspaceDocument | null;
  highlightedBoundingBoxId: string | null;
  activeCitation?: Citation | null;
  onOpenChange: (open: boolean) => void;
}

export function DocumentViewerModal({
  open,
  document,
  highlightedBoundingBoxId,
  activeCitation = null,
  onOpenChange,
}: DocumentViewerModalProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="flex h-[85vh] max-w-5xl flex-col p-0">
        <DialogHeader className="border-b border-zinc-200 px-6 py-4">
          <DialogTitle>{document?.name ?? "Document viewer"}</DialogTitle>
        </DialogHeader>
        <div className="min-h-0 flex-1">
          <CitationViewer
            document={document}
            highlightedBoundingBoxId={highlightedBoundingBoxId}
            activeCitation={activeCitation}
          />
        </div>
      </DialogContent>
    </Dialog>
  );
}
