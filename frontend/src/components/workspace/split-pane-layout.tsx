"use client";

import { useEffect, useState } from "react";
import { BookOpen, PanelLeft } from "lucide-react";

import { DropZone } from "@/components/upload/drop-zone";
import { UploadProgressList } from "@/components/upload/upload-progress-list";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { ChatPanel } from "@/components/workspace/chat-panel";
import { CitationViewer } from "@/components/workspace/citation-viewer";
import { DocumentViewerModal } from "@/components/workspace/document-viewer-modal";
import { SourceManager } from "@/components/workspace/source-manager";
import { useChat } from "@/hooks/use-chat";
import { useUploadPipeline } from "@/hooks/use-upload-pipeline";
import { fetchWorkspaceDocuments } from "@/lib/api-client";
import { useWorkspaceStore } from "@/stores/workspace-store";

interface SplitPaneLayoutProps {
  workspaceId: string;
  workspaceName?: string;
}

export function SplitPaneLayout({
  workspaceId,
  workspaceName = "Document Workspace",
}: SplitPaneLayoutProps) {
  const [showSources, setShowSources] = useState(true);

  const setWorkspaceId = useWorkspaceStore((s) => s.setWorkspaceId);
  const documents = useWorkspaceStore((s) => s.documents);
  const uploadQueue = useWorkspaceStore((s) => s.uploadQueue);
  const activeDocumentId = useWorkspaceStore((s) => s.activeDocumentId);
  const highlightedBoundingBoxId = useWorkspaceStore(
    (s) => s.highlightedBoundingBoxId,
  );
  const activeCitation = useWorkspaceStore((s) => s.activeCitation);
  const viewerOpen = useWorkspaceStore((s) => s.viewerOpen);
  const setDocuments = useWorkspaceStore((s) => s.setDocuments);
  const setActiveDocument = useWorkspaceStore((s) => s.setActiveDocument);
  const toggleDocumentSelection = useWorkspaceStore(
    (s) => s.toggleDocumentSelection,
  );
  const setViewerOpen = useWorkspaceStore((s) => s.setViewerOpen);
  const navigateToCitation = useWorkspaceStore((s) => s.navigateToCitation);

  const { enqueueFiles } = useUploadPipeline(workspaceId);
  const { chatMessages, isChatLoading, sendMessage } = useChat(workspaceId);

  useEffect(() => {
    setWorkspaceId(workspaceId);
  }, [workspaceId, setWorkspaceId]);

  useEffect(() => {
    void fetchWorkspaceDocuments(workspaceId)
      .then(setDocuments)
      .catch(() => {
        // Backend may be unavailable during local dev; uploads still work
      });
  }, [workspaceId, setDocuments]);

  const activeDocument =
    documents.find((doc) => doc.id === activeDocumentId) ?? null;

  return (
    <div className="flex h-screen flex-col bg-zinc-50">
      <header className="flex shrink-0 items-center justify-between border-b border-zinc-200 bg-white px-4 py-3">
        <div className="flex items-center gap-3">
          <BookOpen className="h-5 w-5 text-zinc-700" />
          <div>
            <h1 className="text-sm font-semibold text-zinc-900">
              {workspaceName}
            </h1>
            <p className="text-xs text-zinc-500">NotebookLM-style workspace</p>
          </div>
        </div>
        <Button
          variant="ghost"
          size="sm"
          className="lg:hidden"
          onClick={() => setShowSources((prev) => !prev)}
        >
          <PanelLeft className="h-4 w-4" />
          Sources
        </Button>
      </header>

      <div className="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-[280px_1fr_360px]">
        {/* Left pane — Source Manager */}
        <aside
          className={`flex min-h-0 flex-col border-r border-zinc-200 bg-white ${
            showSources ? "flex" : "hidden lg:flex"
          }`}
        >
          <div className="border-b border-zinc-200 p-4">
            <DropZone onFilesAccepted={enqueueFiles} />
          </div>
          <ScrollArea className="flex-1">
            <div className="p-4">
              <UploadProgressList items={uploadQueue} />
              {uploadQueue.length > 0 && documents.length > 0 && (
                <Separator className="my-4" />
              )}
              <SourceManager
                documents={documents}
                activeDocumentId={activeDocumentId}
                onSelectDocument={setActiveDocument}
                onToggleSelection={toggleDocumentSelection}
              />
            </div>
          </ScrollArea>
        </aside>

        {/* Center pane — Citation Viewer */}
        <main className="hidden min-h-0 flex-col lg:flex">
          <CitationViewer
            document={activeDocument}
            highlightedBoundingBoxId={highlightedBoundingBoxId}
            activeCitation={activeCitation}
          />
        </main>

        {/* Right pane — Chat */}
        <aside className="min-h-0 border-l border-zinc-200">
          <ChatPanel
            messages={chatMessages}
            documents={documents}
            isLoading={isChatLoading}
            onSend={sendMessage}
            onToggleDocument={toggleDocumentSelection}
            onCitationClick={navigateToCitation}
          />
        </aside>
      </div>

      <DocumentViewerModal
        open={viewerOpen}
        document={activeDocument}
        highlightedBoundingBoxId={highlightedBoundingBoxId}
        activeCitation={activeCitation}
        onOpenChange={setViewerOpen}
      />
    </div>
  );
}
