import type { Metadata } from "next";

import { Chat } from "@/components/Chat";

export const metadata: Metadata = { title: "Chat" };

export default function ChatPage() {
  return (
    <section className="flex flex-col gap-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Chat</h1>
        <p className="mt-1 text-sm text-muted">
          Ask the agents. Their tool calls and the documents they cite appear as the answer
          streams in; reminder drafts wait here for your approval.
        </p>
      </div>
      <Chat />
    </section>
  );
}
