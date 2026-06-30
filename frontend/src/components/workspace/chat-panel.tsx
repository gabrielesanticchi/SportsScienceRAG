"use client";

import { FormEvent, useRef, useEffect } from "react";
import { Bot, Send, User } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";
import type { ChatMessage, Citation, WorkspaceDocument } from "@/lib/types";

interface ChatPanelProps {
  messages: ChatMessage[];
  documents: WorkspaceDocument[];
  isLoading: boolean;
  onSend: (message: string) => void;
  onToggleDocument: (id: string) => void;
  onCitationClick: (citation: Citation) => void;
}

function CitationChip({
  citation,
  onClick,
}: {
  citation: Citation;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="inline-flex items-center rounded-full bg-amber-100 px-2 py-0.5 text-xs font-medium text-amber-900 transition-colors hover:bg-amber-200"
    >
      [{citation.label}]
    </button>
  );
}

export function ChatPanel({
  messages,
  documents,
  isLoading,
  onSend,
  onToggleDocument,
  onCitationClick,
}: ChatPanelProps) {
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const selectedCount = documents.filter((d) => d.selected).length;

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    const value = inputRef.current?.value ?? "";
    if (!value.trim() || isLoading) return;
    onSend(value);
    if (inputRef.current) inputRef.current.value = "";
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSubmit(event);
    }
  };

  return (
    <div className="flex h-full flex-col bg-white">
      <div className="border-b border-zinc-200 px-4 py-3">
        <h2 className="text-sm font-semibold text-zinc-900">Notebook</h2>
        <p className="text-xs text-zinc-500">
          {selectedCount} of {documents.length} sources active
        </p>
      </div>

      <div className="border-b border-zinc-200 px-4 py-3">
        <p className="mb-2 text-xs font-medium uppercase tracking-wide text-zinc-500">
          Source filters
        </p>
        <div className="max-h-28 space-y-2 overflow-y-auto">
          {documents.length === 0 ? (
            <p className="text-xs text-zinc-400">Upload documents to chat</p>
          ) : (
            documents.map((doc) => (
              <label
                key={doc.id}
                className="flex cursor-pointer items-center gap-2 text-sm"
              >
                <Checkbox
                  checked={doc.selected}
                  onCheckedChange={() => onToggleDocument(doc.id)}
                />
                <span className="truncate text-zinc-700">{doc.name}</span>
              </label>
            ))
          )}
        </div>
      </div>

      <ScrollArea className="flex-1 px-4">
        <div className="space-y-4 py-4">
          {messages.length === 0 && (
            <div className="rounded-lg bg-zinc-50 p-4 text-center">
              <Bot className="mx-auto mb-2 h-6 w-6 text-zinc-400" />
              <p className="text-sm text-zinc-600">
                Ask questions across your selected sources
              </p>
            </div>
          )}

          {messages.map((message) => (
            <div
              key={message.id}
              className={cn(
                "flex gap-3",
                message.role === "user" ? "flex-row-reverse" : "flex-row",
              )}
            >
              <div
                className={cn(
                  "flex h-7 w-7 shrink-0 items-center justify-center rounded-full",
                  message.role === "user"
                    ? "bg-zinc-900 text-white"
                    : "bg-zinc-100 text-zinc-600",
                )}
              >
                {message.role === "user" ? (
                  <User className="h-3.5 w-3.5" />
                ) : (
                  <Bot className="h-3.5 w-3.5" />
                )}
              </div>
              <div
                className={cn(
                  "max-w-[85%] rounded-2xl px-3 py-2 text-sm leading-relaxed",
                  message.role === "user"
                    ? "bg-zinc-900 text-white"
                    : "bg-zinc-100 text-zinc-900",
                )}
              >
                <p className="whitespace-pre-wrap">{message.content}</p>
                {message.citations && message.citations.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    {message.citations.map((citation) => (
                      <CitationChip
                        key={citation.id}
                        citation={citation}
                        onClick={() => onCitationClick(citation)}
                      />
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="flex gap-3">
              <div className="flex h-7 w-7 items-center justify-center rounded-full bg-zinc-100">
                <Bot className="h-3.5 w-3.5 text-zinc-600" />
              </div>
              <div className="rounded-2xl bg-zinc-100 px-3 py-2 text-sm text-zinc-500">
                Thinking…
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>
      </ScrollArea>

      <Separator />
      <form onSubmit={handleSubmit} className="p-4">
        <div className="flex gap-2">
          <textarea
            ref={inputRef}
            rows={2}
            placeholder="Ask about your sources…"
            disabled={isLoading || selectedCount === 0}
            onKeyDown={handleKeyDown}
            className="flex-1 resize-none rounded-lg border border-zinc-200 px-3 py-2 text-sm outline-none focus:border-zinc-400 disabled:opacity-50"
          />
          <Button
            type="submit"
            size="icon"
            disabled={isLoading || selectedCount === 0}
            aria-label="Send message"
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </form>
    </div>
  );
}
