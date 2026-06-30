"use client";

import { useCallback } from "react";
import { v4 as uuidv4 } from "uuid";

import { submitSearchQuery } from "@/lib/api-client";
import type { ChatMessage } from "@/lib/types";
import {
  getSelectedDocumentIds,
  useWorkspaceStore,
} from "@/stores/workspace-store";

export function useChat(workspaceId: string) {
  const documents = useWorkspaceStore((s) => s.documents);
  const chatMessages = useWorkspaceStore((s) => s.chatMessages);
  const isChatLoading = useWorkspaceStore((s) => s.isChatLoading);
  const addChatMessage = useWorkspaceStore((s) => s.addChatMessage);
  const setChatLoading = useWorkspaceStore((s) => s.setChatLoading);

  const sendMessage = useCallback(
    async (content: string) => {
      const trimmed = content.trim();
      if (!trimmed) return;

      const userMessage: ChatMessage = {
        id: uuidv4(),
        role: "user",
        content: trimmed,
        timestamp: new Date().toISOString(),
      };
      addChatMessage(userMessage);
      setChatLoading(true);

      try {
        const documentIds = getSelectedDocumentIds(documents);
        const { answer, citations } = await submitSearchQuery({
          query: trimmed,
          workspace_id: workspaceId,
          document_ids: documentIds,
        });

        const assistantMessage: ChatMessage = {
          id: uuidv4(),
          role: "assistant",
          content: answer,
          citations,
          timestamp: new Date().toISOString(),
        };
        addChatMessage(assistantMessage);
      } catch (error) {
        const assistantMessage: ChatMessage = {
          id: uuidv4(),
          role: "assistant",
          content:
            error instanceof Error
              ? `Sorry, something went wrong: ${error.message}`
              : "Sorry, something went wrong.",
          timestamp: new Date().toISOString(),
        };
        addChatMessage(assistantMessage);
      } finally {
        setChatLoading(false);
      }
    },
    [
      workspaceId,
      documents,
      addChatMessage,
      setChatLoading,
    ],
  );

  return { chatMessages, isChatLoading, sendMessage };
}
