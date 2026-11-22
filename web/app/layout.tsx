import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";

import { Sidebar } from "@/components/Sidebar";
import { THEME_SCRIPT } from "@/components/theme";
import { ThemeToggle } from "@/components/ThemeToggle";

import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: { default: "LedgerLens", template: "%s · LedgerLens" },
  description: "Accounts-payable ledger: documents, questions, anomalies and reminders.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    // The theme script sets data-theme before React hydrates.
    <html
      lang="en"
      suppressHydrationWarning
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body className="flex min-h-full flex-col font-sans md:flex-row">
        <Sidebar footer={<ThemeToggle />} />
        <main className="min-w-0 flex-1 px-6 py-8 md:h-screen md:overflow-y-auto md:px-10">
          <div className="mx-auto max-w-6xl">{children}</div>
        </main>
      </body>
    </html>
  );
}
