import type { Metadata } from "next";
import "./globals.css";
import { Sidebar } from "@/components/sidebar";

export const metadata: Metadata = {
  title: "rPPG Results — camera-only heart rate",
  description:
    "Validation of PHASE-Net and a classical POS baseline against UBFC-rPPG, with a controlled motion protocol. KEIKO project, TU Clausthal.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="flex min-h-screen">
          <Sidebar />
          <div className="min-w-0 flex-1 bg-grid-fade">
            <main className="mx-auto max-w-[1400px] space-y-6 px-5 py-6 lg:px-8">
              {children}
            </main>
          </div>
        </div>
      </body>
    </html>
  );
}
