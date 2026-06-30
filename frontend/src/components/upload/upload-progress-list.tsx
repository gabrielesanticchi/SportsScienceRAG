"use client";

import { AlertCircle, CheckCircle2, Loader2 } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { statusLabel } from "@/hooks/use-upload-pipeline";
import { formatFileSize } from "@/lib/utils";
import type { UploadFileItem } from "@/lib/types";

interface UploadProgressListProps {
  items: UploadFileItem[];
}

function statusVariant(
  status: UploadFileItem["status"],
): "default" | "secondary" | "success" | "warning" | "destructive" {
  switch (status) {
    case "COMPLETED":
      return "success";
    case "FAILED":
      return "destructive";
    case "PROCESSING":
      return "warning";
    default:
      return "secondary";
  }
}

export function UploadProgressList({ items }: UploadProgressListProps) {
  if (items.length === 0) return null;

  return (
    <div className="space-y-3">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-zinc-500">
        Upload queue
      </h3>
      <ul className="space-y-2">
        {items.map((item) => (
          <li
            key={item.id}
            className="rounded-lg border border-zinc-200 bg-white p-3 shadow-sm"
          >
            <div className="mb-2 flex items-start justify-between gap-2">
              <div className="min-w-0">
                <p className="truncate text-sm font-medium text-zinc-900">
                  {item.name}
                </p>
                <p className="text-xs text-zinc-500">
                  {formatFileSize(item.size)}
                </p>
              </div>
              <Badge variant={statusVariant(item.status)}>
                {item.status}
              </Badge>
            </div>

            {(item.status === "UPLOADING" || item.status === "PROCESSING") && (
              <Progress value={item.progress} className="mb-2" />
            )}

            <div className="flex items-center gap-2 text-xs text-zinc-600">
              {item.status === "FAILED" ? (
                <AlertCircle className="h-3.5 w-3.5 shrink-0 text-red-500" />
              ) : item.status === "COMPLETED" ? (
                <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-emerald-500" />
              ) : (
                <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin" />
              )}
              <span>{statusLabel(item.status, item.subState)}</span>
            </div>

            {item.error && (
              <p className="mt-2 text-xs text-red-600">{item.error}</p>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
