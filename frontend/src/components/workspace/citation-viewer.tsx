"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { getAuthToken } from "@/lib/auth";
import { cn } from "@/lib/utils";
import type { BoundingBox, Citation, WorkspaceDocument } from "@/lib/types";

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
  activeCitation?: Citation | null;
}

function hasLayoutOverlay(box: BoundingBox): boolean {
  return !box.estimated && box.width > 0 && box.height > 0;
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
  activeCitation = null,
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
    const targetPage =
      activeCitation?.page ??
      document?.boundingBoxes.find((box) => box.id === highlightedBoundingBoxId)
        ?.page;

    if (!targetPage) return;

    const pageEl = pageRefs.current.get(targetPage);
    pageEl?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [activeCitation, highlightedBoundingBoxId, document, numPages]);

  const layoutBoxes =
    document?.boundingBoxes.filter((box) => hasLayoutOverlay(box)) ?? [];

  const citationPreview =
    activeCitation?.text ??
    document?.boundingBoxes.find((box) => box.id === highlightedBoundingBoxId)
      ?.text;

  const onDocumentLoadSuccess = useCallback(({ numPages: pages }: { numPages: number }) => {
    setNumPages(pages);
  }, []);

  const pdfFile = useMemo(() => {
    if (!document?.pdfUrl) return null;
    if (document.pdfUrl.startsWith("blob:")) {
      return document.pdfUrl;
    }

    const token = getAuthToken();
    if (!token) {
      return document.pdfUrl;
    }

    return {
      url: document.pdfUrl,
      httpHeaders: {
        Authorization: `Bearer ${token}`,
      },
    };
  }, [document?.pdfUrl]);

  if (!document) {
    return (
      <div className="flex h-full items-center justify-center bg-zinc-50">
        <p className="text-sm text-zinc-500">
          Select a source document to view citations
        </p>
      </div>
    );
  }

  if (!pdfFile) {
    return (
      <div className="flex h-full items-center justify-center bg-zinc-50">
        <p className="text-sm text-zinc-500">PDF preview unavailable</p>
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col bg-zinc-100">
      {activeCitation && citationPreview && (
        <div className="border-b border-amber-200 bg-amber-50 px-4 py-3">
          <p className="text-xs font-semibold uppercase tracking-wide text-amber-800">
            Citation [{activeCitation.label}] · Page {activeCitation.page}
          </p>
          <p className="mt-1 text-sm leading-relaxed text-amber-950">
            {citationPreview}
          </p>
        </div>
      )}
      <div ref={containerRef} className="min-h-0 flex-1 overflow-y-auto p-4">
      <Document
        file={pdfFile}
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
              {layoutBoxes.length > 0 && (
                <BoundingBoxOverlay
                  boxes={layoutBoxes}
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
    </div>
  );
}
