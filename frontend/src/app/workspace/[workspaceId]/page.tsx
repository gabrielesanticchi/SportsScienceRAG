"use client";

import { SplitPaneLayout } from "@/components/workspace/split-pane-layout";
import { useAuth } from "@/components/providers/auth-provider";
import { use } from "react";

interface WorkspacePageProps {
  params: Promise<{ workspaceId: string }>;
}

export default function WorkspacePage({ params }: WorkspacePageProps) {
  const { workspaceId } = use(params);
  const { isReady } = useAuth();

  if (!isReady) {
    return (
      <div className="flex h-screen items-center justify-center text-sm text-zinc-500">
        Initializing session…
      </div>
    );
  }

  return (
    <SplitPaneLayout
      workspaceId={workspaceId}
      workspaceName={`Workspace · ${workspaceId}`}
    />
  );
}
