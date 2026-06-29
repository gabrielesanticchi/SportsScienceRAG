"use client";

import { useCallback, useRef, useState } from "react";
import { FileUp, Upload } from "lucide-react";

import { Button } from "@/components/ui/button";
import { cn, formatFileSize } from "@/lib/utils";
import { MAX_PDF_SIZE_BYTES } from "@/lib/types";

interface DropZoneProps {
  onFilesAccepted: (files: FileList | File[]) => void;
  disabled?: boolean;
}

export function DropZone({ onFilesAccepted, disabled }: DropZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.stopPropagation();
    if (!disabled) setIsDragging(true);
  }, [disabled]);

  const handleDragLeave = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      event.stopPropagation();
      setIsDragging(false);
      if (disabled || !event.dataTransfer.files.length) return;
      onFilesAccepted(event.dataTransfer.files);
    },
    [disabled, onFilesAccepted],
  );

  const handleFileInput = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      if (event.target.files?.length) {
        onFilesAccepted(event.target.files);
        event.target.value = "";
      }
    },
    [onFilesAccepted],
  );

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={cn(
        "relative flex flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-10 transition-colors",
        isDragging
          ? "border-zinc-900 bg-zinc-50"
          : "border-zinc-200 bg-white hover:border-zinc-300",
        disabled && "pointer-events-none opacity-50",
      )}
    >
      <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-zinc-100">
        <Upload className="h-5 w-5 text-zinc-600" />
      </div>
      <p className="mb-1 text-sm font-medium text-zinc-900">
        Drop PDF sources here
      </p>
      <p className="mb-4 text-xs text-zinc-500">
        PDF only · up to {formatFileSize(MAX_PDF_SIZE_BYTES)} per file
      </p>
      <Button
        type="button"
        variant="secondary"
        size="sm"
        onClick={() => inputRef.current?.click()}
      >
        <FileUp className="h-4 w-4" />
        Browse files
      </Button>
      <input
        ref={inputRef}
        type="file"
        accept="application/pdf,.pdf"
        multiple
        className="hidden"
        onChange={handleFileInput}
      />
    </div>
  );
}
