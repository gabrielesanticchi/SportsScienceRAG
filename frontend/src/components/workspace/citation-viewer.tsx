"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useRef, useState } from "react";

import { cn } from "@/lib/utils";
import type { BoundingBox, WorkspaceDocument } from "@/lib/types";

const Document = dynamic(
  () => import("react-pdf").then((mod) => mod.Document),
  { ssr: false },
);

const Page = dynamic(() => import("react-pdf").then((mod) => mod.Page), {
  ssr: false,
});

interface CitationViewerProps {
  document: WorkspaceDocument | null;
  highlightedBoundingBoxId: string | null;
}

const BOX_COLORS: Record<BoundingBox["type"], string> = {
  paragraph: "rgba(59, 130, 246, 0.25)",
  header: "rgba(168, 85, 247, 0.25)",
  table: "rgba(16, 185, 129, 0.25)",
};

const BOX_BORDER: Record<BoundingBox["type"], string> = {
  paragraph: "rgba(59, 130, 246, 0.8)",
  header: "rgba(168, 85, 247, 0.8)",
  table: "rgba(16, 185, 129, 0.8)",
};

function BoundingBoxOverlay({
  boxes,
  pageNumber,
  pageWidth,
  pageHeight,
  highlightedId,
}: {
  boxes: BoundingBox[];
  pageNumber: number;
  pageWidth: number;
  pageHeight: number;
  highlightedId: string | null;
}) {
  const pageBoxes = boxes.filter((box) => box.page === pageNumber);

  return (
    <svg
      className="pointer-events-none absolute inset-0 h-full w-full"
      viewBox={`0 0 ${pageWidth} ${pageHeight}`}
      preserveAspectRatio="none"
    >
      {pageBoxes.map((box) => {
        const isHighlighted = box.id === highlightedId;
        return (
          <rect
            key={box.id}
            x={box.x}
            y={box.y}
            width={box.width}
            height={box.height}
            fill={isHighlighted ? "rgba(250, 204, 21, 0.45)" : BOX_COLORS[box.type]}
            stroke={isHighlighted ? "#ca8a04" : BOX_BORDER[box.type]}
            strokeWidth={isHighlighted ? 3 : 1.5}
            rx={2}
          />
        );
      })}
    </svg>
  );
}

export function CitationViewer({
  document,
  highlightedBoundingBoxId,
}: CitationViewerProps) {
  const [numPages, setNumPages] = useState(0);
  const [pageWidth, setPageWidth] = useState(600);
  const containerRef = useRef<HTMLDivElement>(null);
  const pageRefs = useRef<Map<number, HTMLDivElement>>(new Map());

  useEffect(() => {
    void import("react-pdf").then(({ pdfjs }) => {
      pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;
    });
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const observer = new ResizeObserver((entries) => {
      const width = entries[0]?.contentRect.width;
      if (width) setPageWidth(Math.min(width - 32, 800));
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!highlightedBoundingBoxId || !document) return;

    const box = document.boundingBoxes.find(
      (b) => b.id === highlightedBoundingBoxId,
    );
    if (!box) return;

    const pageEl = pageRefs.current.get(box.page);
    pageEl?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [highlightedBoundingBoxId, document]);

  const onDocumentLoadSuccess = useCallback(({ numPages: pages }: { numPages: number }) => {
    setNumPages(pages);
  }, []);

  if (!document) {
    return (
      <div className="flex h-full items-center justify-center bg-zinc-50">
        <p className="text-sm text-zinc-500">
          Select a source document to view citations
        </p>
      </div>
    );
  }

  if (!document.pdfUrl) {
    return (
      <div className="flex h-full items-center justify-center bg-zinc-50">
        <p className="text-sm text-zinc-500">PDF preview unavailable</p>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="h-full overflow-y-auto bg-zinc-100 p-4">
      <Document
        file={document.pdfUrl}
        onLoadSuccess={onDocumentLoadSuccess}
        loading={
          <div className="flex h-40 items-center justify-center text-sm text-zinc-500">
            Loading PDF…
          </div>
        }
        error={
          <div className="flex h-40 items-center justify-center text-sm text-red-500">
            Failed to load PDF
          </div>
        }
      >
        {Array.from({ length: numPages }, (_, index) => {
          const pageNumber = index + 1;
          const aspectRatio = 1.414;
          const pageHeight = pageWidth * aspectRatio;

          return (
            <div
              key={pageNumber}
              ref={(el) => {
                if (el) pageRefs.current.set(pageNumber, el);
              }}
              className={cn(
                "relative mx-auto mb-6 shadow-md",
                "w-fit",
              )}
            >
              <Page
                pageNumber={pageNumber}
                width={pageWidth}
                renderTextLayer={false}
                renderAnnotationLayer={false}
              />
              {document.boundingBoxes.length > 0 && (
                <BoundingBoxOverlay
                  boxes={document.boundingBoxes}
                  pageNumber={pageNumber}
                  pageWidth={pageWidth}
                  pageHeight={pageHeight}
                  highlightedId={highlightedBoundingBoxId}
                />
              )}
            </div>
          );
        })}
      </Document>
    </div>
  );
}
