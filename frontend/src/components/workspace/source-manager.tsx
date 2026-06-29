"use client";

import { FileText, Loader2 } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { cn } from "@/lib/utils";
import type { WorkspaceDocument } from "@/lib/types";

interface SourceManagerProps {
  documents: WorkspaceDocument[];
  activeDocumentId: string | null;
  onSelectDocument: (id: string) => void;
  onToggleSelection: (id: string) => void;
}

function statusBadgeVariant(
  status: WorkspaceDocument["status"],
): "success" | "warning" | "destructive" | "secondary" {
  switch (status) {
    case "COMPLETED":
      return "success";
    case "PROCESSING":
    case "UPLOADING":
      return "warning";
    case "FAILED":
      return "destructive";
    default:
      return "secondary";
  }
}

export function SourceManager({
  documents,
  activeDocumentId,
  onSelectDocument,
  onToggleSelection,
}: SourceManagerProps) {
  if (documents.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center px-4 text-center">
        <FileText className="mb-3 h-8 w-8 text-zinc-300" />
        <p className="text-sm text-zinc-500">No sources yet</p>
        <p className="mt-1 text-xs text-zinc-400">
          Upload PDFs to build your workspace
        </p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 gap-3 p-4 sm:grid-cols-1 xl:grid-cols-2">
      {documents.map((doc) => {
        const isActive = doc.id === activeDocumentId;
        const isProcessing =
          doc.status === "PROCESSING" || doc.status === "UPLOADING";

        return (
          <div
            key={doc.id}
            className={cn(
              "group relative flex flex-col rounded-xl border bg-white p-3 shadow-sm transition-all hover:shadow-md",
              isActive
                ? "border-zinc-900 ring-1 ring-zinc-900"
                : "border-zinc-200",
            )}
          >
            <div className="mb-3 flex items-start justify-between gap-2">
              <Checkbox
                checked={doc.selected}
                onCheckedChange={() => onToggleSelection(doc.id)}
                aria-label={`Include ${doc.name} in chat context`}
              />
              <Badge variant={statusBadgeVariant(doc.status)} className="shrink-0">
                {doc.status}
              </Badge>
            </div>

            <button
              type="button"
              onClick={() => onSelectDocument(doc.id)}
              className="flex flex-1 flex-col items-start text-left"
            >
              <div className="mb-2 flex h-20 w-full items-center justify-center rounded-lg bg-zinc-50">
                {isProcessing ? (
                  <Loader2 className="h-6 w-6 animate-spin text-zinc-400" />
                ) : (
                  <FileText className="h-8 w-8 text-zinc-400" />
                )}
              </div>
              <p className="line-clamp-2 text-sm font-medium text-zinc-900">
                {doc.name}
              </p>
              {doc.subState && doc.status === "PROCESSING" && (
                <p className="mt-1 text-xs text-amber-700">{doc.subState}</p>
              )}
              {doc.pageCount && (
                <p className="mt-1 text-xs text-zinc-500">
                  {doc.pageCount} pages
                </p>
              )}
            </button>
          </div>
        );
      })}
    </div>
  );
}
